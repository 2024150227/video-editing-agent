import json
import logging
import httpx
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """火山方舟 LLM API 客户端"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.VOLCANO_ARK_API_KEY
        self.base_url = base_url or settings.VOLCANO_ARK_BASE_URL
        self.model = model or settings.VOLCANO_ARK_MODEL

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
            self,
            messages: list[dict],
            temperature: float = 0.7,
            max_tokens: int = 4096,
    ) -> str:
        """发送对话请求，返回文本响应"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_json(
            self,
            messages: list[dict],
            temperature: float = 0.7,
            max_tokens: int = 4096,
            retries: int = 3,
    ) -> dict:
        """发送请求并强制解析 JSON 返回，解析失败自动重试"""
        messages = [*messages]
        last_raw = ""
        for attempt in range(retries):
            raw = await self.chat(messages, temperature, max_tokens)
            last_raw = raw
            try:
                # 尝试提取 JSON 块
                text = raw.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return json.loads(text)
            except (json.JSONDecodeError, IndexError):
                logger.warning(f"JSON parse failed, attempt {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    messages.append({"role": "user", "content": "请严格按照 JSON 格式返回，不要包含其他文字。"})

        return {"raw": last_raw, "error": f"JSON parse failed after {retries} retries"}
