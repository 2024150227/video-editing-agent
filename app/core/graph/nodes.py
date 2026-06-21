"""LangGraph Nodes — 导演 / 素材 / 剪辑 三个核心节点"""

import os
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import interrupt

from app.core.graph.state import VideoEditingState
from app.core.graph.llm import get_chat_model
from app.core.graph.tools import search_materials, generate_material
from app.services.render_service import RenderService

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# 导演节点 System Prompt
# ──────────────────────────────────────────────────────────
DIRECTOR_SYSTEM_PROMPT = """你是一个专业的视频导演。根据用户的创意描述，生成详细的分镜脚本。
你必须严格返回 JSON 格式，不包含其他文字：

{
  "style": "视频风格描述",
  "total_duration_sec": 30,
  "bgm_mood": "BGM情绪描述",
  "shots": [
    {
      "index": 1,
      "duration_sec": 3,
      "description": "画面内容描述（具体、可用于素材检索）",
      "shot_type": "wide|medium|close-up|detail",
      "camera_motion": "静态|推|拉|摇|移|跟",
      "transition": "硬切|淡入|淡出|叠化",
      "mood_words": ["情绪词1", "情绪词2"]
    }
  ]
}

要求：
1. 所有镜头的 duration_sec 之和必须精确等于 total_duration_sec
2. 快节奏视频镜头时长短（2-4秒），慢节奏镜头时长长（5-8秒）
3. 每个镜头描述独立、具体"""

# ──────────────────────────────────────────────────────────
# 导演节点
# ──────────────────────────────────────────────────────────
async def director_node(state: VideoEditingState) -> dict:
    """根据用户 prompt 生成分镜脚本，然后暂停等待人工确认"""
    logger.info("[Director] 开始生成分镜脚本...")

    llm = get_chat_model()
    parser = JsonOutputParser()

    messages = [
        SystemMessage(content=DIRECTOR_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"创意描述：{state['prompt']}\n"
            f"目标风格：{state['style']}\n"
            f"要求总时长：{state['total_duration_sec']} 秒\n"
            f"请生成分镜脚本。"
        )),
    ]

    # LLM 调用，强制 JSON 输出
    response = await llm.ainvoke(messages)

    # 解析 JSON（带重试）
    storyboard = _parse_json_with_fallback(response.content)

    # 规则校验：时长检查
    shots = storyboard.get("shots", [])
    actual_duration = sum(s.get("duration_sec", 0) for s in shots)
    target = state["total_duration_sec"]
    if abs(actual_duration - target) > 1 and shots:
        shots[-1]["duration_sec"] = max(1, shots[-1]["duration_sec"] + target - actual_duration)
        storyboard["shots"] = shots
        storyboard["total_duration_sec"] = target

    # interrupt: 暂停等待用户确认分镜
    user_review = interrupt({
        "step": "storyboard_review",
        "storyboard": storyboard,
        "message": "请审核分镜脚本。修改后发送确认以继续。",
    })

    # 用户可能修改了分镜，使用修改后的版本
    if isinstance(user_review, dict) and "storyboard" in user_review:
        storyboard = user_review["storyboard"]

    return {
        "storyboard": storyboard,
        "current_step": "storyboard_approved",
        "messages": [SystemMessage(content="分镜脚本已确认")],
    }


# ──────────────────────────────────────────────────────────
# 素材节点
# ──────────────────────────────────────────────────────────
async def material_node(state: VideoEditingState) -> dict:
    """为每个镜头匹配素材，然后暂停等待用户选择"""
    logger.info("[Material] 开始匹配素材...")

    mode = state.get("mode", "retrieval")
    storyboard = state["storyboard"]
    overrides = state.get("material_overrides", {}) or {}
    all_matches = []

    for shot in storyboard["shots"]:
        shot_index = str(shot["index"])

        # 用户指定特定文件则跳过检索
        if shot_index in overrides:
            all_matches.append({
                "shot_index": shot["index"],
                "candidates": [{
                    "material_id": f"override_{shot_index}",
                    "file_name": overrides[shot_index],
                    "clip_range": [0, shot["duration_sec"]],
                    "score": 1.0,
                    "reason": "用户指定",
                }],
            })
            continue

        if mode == "retrieval":
            raw = search_materials.invoke({
                "description": shot["description"],
                "top_k": 5,
            })
            candidates = json.loads(raw) if isinstance(raw, str) else raw
            top_candidates = [
                {
                    "material_id": c.get("point_id", ""),
                    "file_name": c.get("file_name", ""),
                    "clip_range": [0, shot["duration_sec"]],
                    "score": round(c.get("score", 0), 4),
                    "reason": f"语义匹配得分 {c.get('score', 0):.2f}",
                }
                for c in candidates[:3]
            ]
        else:
            enhanced = generate_material.invoke({"description": shot["description"]})
            top_candidates = [{
                "material_id": f"gen_{shot['index']}",
                "file_name": f"generated_shot_{shot['index']}.png",
                "clip_range": [0, shot["duration_sec"]],
                "score": 1.0,
                "reason": "AI 生成素材",
                "generation_prompt": enhanced,
            }]

        all_matches.append({
            "shot_index": shot["index"],
            "candidates": top_candidates,
        })

    # interrupt: 暂停等待用户选择素材
    user_selections = interrupt({
        "step": "material_review",
        "material_matches": all_matches,
        "message": "请为每个镜头选择素材。返回 material_selections 以继续。",
    })

    return {
        "material_matches": all_matches,
        "material_selections": user_selections if isinstance(user_selections, list) else [],
        "current_step": "materials_approved",
        "messages": [SystemMessage(content="素材已选择")],
    }


# ──────────────────────────────────────────────────────────
# 剪辑节点
# ──────────────────────────────────────────────────────────
async def editor_node(state: VideoEditingState) -> dict:
    """同步渲染视频（在 executor 中运行以释放事件循环）"""
    import asyncio
    import uuid

    project_id = state["project_id"]
    storyboard = state["storyboard"]
    material_selections = state.get("material_selections", [])
    bgm_path = state.get("bgm_path")

    task_id = str(uuid.uuid4())

    logger.info(f"[Editor] 开始渲染: {task_id}")

    def _render():
        service = RenderService()
        clip = service.build_timeline(storyboard, material_selections, bgm_path)
        output_filename = f"{project_id}_{task_id}.mp4"
        output_path = service.render(clip, output_filename)
        if bgm_path and os.path.exists(str(bgm_path)):
            output_path = service.add_bgm(output_path, bgm_path)
        clip.close()
        return output_path

    loop = asyncio.get_event_loop()
    try:
        output_path = await loop.run_in_executor(None, _render)
        logger.info(f"[Editor] 渲染完成: {output_path}")
        return {
            "task_id": task_id,
            "task_status": {"status": "completed", "progress_pct": 100, "output_path": output_path},
            "current_step": "done",
            "messages": [SystemMessage(content=f"渲染完成: {output_path}")],
        }
    except Exception as e:
        logger.exception(f"[Editor] 渲染失败")
        return {
            "task_id": task_id,
            "task_status": {"status": "failed", "error": str(e), "progress_pct": 0},
            "current_step": "error",
            "error": str(e),
            "messages": [SystemMessage(content=f"渲染失败: {e}")],
        }


# ──────────────────────────────────────────────────────────
# JSON 解析工具
# ──────────────────────────────────────────────────────────
def _parse_json_with_fallback(text: str) -> dict:
    """从 LLM 输出中提取 JSON，支持 ```json 包裹"""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON 解析失败，使用 fallback")
        return {"raw": text, "error": "JSON parse failed"}
