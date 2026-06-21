"""LangChain ChatModel — 火山方舟适配（OpenAI 兼容接口）"""

from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.core.config import get_settings


@lru_cache()
def get_chat_model() -> ChatOpenAI:
    """返回 LangChain ChatModel，指向火山方舟 API"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.VOLCANO_ARK_MODEL,
        api_key=settings.VOLCANO_ARK_API_KEY,
        base_url=settings.VOLCANO_ARK_BASE_URL,
        temperature=0.7,
        max_tokens=4096,
    )
