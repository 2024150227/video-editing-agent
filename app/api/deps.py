from functools import lru_cache
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.graph.builder import build_graph


@lru_cache()
def get_graph():
    """返回编译后的 LangGraph（单例，共享 MemorySaver）"""
    return build_graph()


async def get_db_session(db: AsyncSession = Depends(get_db)):
    return db
