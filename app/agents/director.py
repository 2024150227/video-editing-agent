import json
import logging
from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

DIRECTOR_SYSTEM_PROMPT = """你是一个专业的视频导演。根据用户的创意描述，生成详细的分镜脚本。
你必须严格返回 JSON 格式，包含以下字段：
- style: 视频风格描述
- total_duration_sec: 总时长（秒）
- bgm_mood: BGM情绪描述
- shots: 分镜列表，每个分镜包含：
  - index: 镜头序号（从1开始）
  - duration_sec: 该镜头时长（秒）
  - description: 画面内容描述（一句话，用于素材检索）
  - shot_type: 景别（wide/medium/close-up/detail）
  - camera_motion: 运镜方式（静态/推/拉/摇/移/跟/升/降）
  - transition: 转场（硬切/淡入/淡出/叠化/闪白）
  - mood_words: 该镜头的情绪关键词列表

要求：
1. 所有镜头的 duration_sec 之和必须精确等于 total_duration_sec
2. 镜头描述要具体，包含主体、动作、场景、光线等信息
3. 节奏与风格匹配：快节奏视频镜头时长短（2-4秒），慢节奏镜头时长长（5-8秒）
4. 每个镜头描述独立，可直接用于素材检索"""


class DirectorAgent:
    """导演 Agent — 将用户创意描述转化为结构化分镜脚本"""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    async def generate_storyboard(
        self,
        prompt: str,
        style: str = "默认风格",
        total_duration_sec: int = 30,
    ) -> dict:
        """生成分镜脚本并校验"""

        messages = self._build_user_message(prompt, style, total_duration_sec)

        # Step 1: 生成分镜
        storyboard = await self.llm.chat_json(
            messages=[{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, messages],
            temperature=0.8,
            max_tokens=4096,
            retries=3,
        )

        # Step 2: 规则校验
        storyboard = self._validate_and_fix(storyboard, total_duration_sec)

        return storyboard

    def _build_user_message(self, prompt: str, style: str, total_duration_sec: int) -> dict:
        return {
            "role": "user",
            "content": f"创意描述：{prompt}\n目标风格：{style}\n要求总时长：{total_duration_sec} 秒\n请生成分镜脚本。",
        }

    def _validate_and_fix(self, storyboard: dict, expected_duration: int) -> dict:
        """校验分镜脚本，并在出错时尝试修复"""
        if "raw" in storyboard:
            return storyboard  # LLM 解析已经失败

        shots = storyboard.get("shots", [])
        actual_duration = sum(s.get("duration_sec", 0) for s in shots)

        # 校验时长
        if abs(actual_duration - expected_duration) > 1:
            logger.warning(
                f"Duration mismatch: expected {expected_duration}s, got {actual_duration}s"
            )
            # 尝试调整最后一个镜头的时长来补齐
            if shots and actual_duration > 0:
                diff = expected_duration - actual_duration
                shots[-1]["duration_sec"] = max(1, shots[-1]["duration_sec"] + diff)
                storyboard["shots"] = shots
                storyboard["total_duration_sec"] = expected_duration
                logger.info(f"Fixed duration: adjusted last shot by {diff}s")

        # 校验每个镜头必填字段
        required_fields = ["index", "duration_sec", "description", "shot_type", "transition", "mood_words"]
        for shot in shots:
            for field in required_fields:
                if field not in shot:
                    shot[field] = [] if field == "mood_words" else ""

        storyboard["total_duration_sec"] = expected_duration
        return storyboard
