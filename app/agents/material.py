import logging
import uuid
from app.core.llm_client import LLMClient
from app.services.index_service import IndexService

logger = logging.getLogger(__name__)

MATERIAL_RANK_PROMPT = """你是一个视频素材匹配专家。根据分镜描述，从候选素材中选出最匹配的 3-5 个。
对每个候选，给出：
- material_id: 素材唯一标识
- file_name: 文件名
- clip_range: 建议裁剪片段 [start_sec, end_sec]，基于镜头时长
- score: 匹配度 0-1
- reason: 匹配理由（一句话中文）

返回 JSON 格式。"""

GENERATION_PROMPT_TEMPLATE = """你是一个视频创作助手。请根据以下镜头描述，扩写为适合 AI 图片/视频生成的 Prompt。
要求：
1. 补充构图方式、光影条件、色彩风格
2. 保持画面主体清晰
3. 风格：电影感、高画质
4. 仅返回扩写后的 Prompt，不要其他内容

镜头描述：{description}"""


class MaterialAgent:
    """素材 Agent — 跨模态检索 + AI 生成"""

    def __init__(self, llm_client: LLMClient | None = None, index_service: IndexService | None = None):
        self.llm = llm_client or LLMClient()
        self.index = index_service or IndexService()

    def match_materials_retrieval(
        self,
        storyboard: dict,
        material_overrides: dict | None = None,
    ) -> list[dict]:
        """检索模式：向量搜索 + LLM 精排"""
        overrides = material_overrides or {}
        results = []

        for shot in storyboard["shots"]:
            shot_index = shot["index"]

            # 用户指定了特定文件则跳过检索
            if str(shot_index) in overrides:
                results.append({
                    "shot_index": shot_index,
                    "candidates": [{
                        "material_id": str(uuid.uuid4()),
                        "file_name": overrides[str(shot_index)],
                        "clip_range": [0, shot["duration_sec"]],
                        "score": 1.0,
                        "reason": "用户指定素材",
                    }],
                })
                continue

            # 粗排：向量检索 Top-10
            candidates = self.index.search_by_text(shot["description"], top_k=10)

            if not candidates:
                results.append({
                    "shot_index": shot_index,
                    "candidates": [],
                    "suggestion": "未找到匹配素材，建议降低匹配阈值或切换到 AI 生成模式",
                })
                continue

            # 精排方案：当前直接使用粗排 Top-3（LLM 精排调用可后续接入）
            top_candidates = []
            for c in candidates[:3]:
                top_candidates.append({
                    "material_id": c["point_id"],
                    "file_name": c["file_name"],
                    "clip_range": [0.0, float(shot["duration_sec"])],
                    "score": round(c["score"], 4),
                    "reason": f"语义匹配得分 {c['score']:.2f}",
                })

            results.append({
                "shot_index": shot_index,
                "candidates": top_candidates,
            })

        return results

    def match_materials_generation(self, storyboard: dict) -> list[dict]:
        """生成模式：为每个镜头生成扩写 Prompt（实际调用生成 API 在后续任务接入）"""
        results = []

        for shot in storyboard["shots"]:
            enhanced_prompt = GENERATION_PROMPT_TEMPLATE.format(
                description=shot["description"]
            )
            results.append({
                "shot_index": shot["index"],
                "candidates": [{
                    "material_id": str(uuid.uuid4()),
                    "file_name": f"generated_shot_{shot['index']}.png",
                    "clip_range": [0, shot["duration_sec"]],
                    "score": 1.0,
                    "reason": "AI 生成素材",
                    "generation_prompt": enhanced_prompt,
                }],
            })

        return results

    async def match_materials(
        self,
        storyboard: dict,
        mode: str = "retrieval",
        material_overrides: dict | None = None,
    ) -> list[dict]:
        """统一入口：根据模式调用检索或生成"""
        if mode == "generation":
            return self.match_materials_generation(storyboard)
        return self.match_materials_retrieval(storyboard, material_overrides)
