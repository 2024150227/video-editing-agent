"""LangChain ChatModel — AgnesAPI 适配（OpenAI 兼容接口）"""

from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.core.config import get_settings


@lru_cache()
def get_chat_model() -> ChatOpenAI:
    """返回 LangChain ChatModel，指向 AgnesAPI"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.AGNES_MODEL,
        api_key=settings.AGNES_API_KEY,
        base_url=settings.AGNES_BASE_URL,
        temperature=0.7,
        max_tokens=4096,
    )
