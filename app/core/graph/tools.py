"""LangGraph Tools — Agent 可调用的工具"""

import json
import logging
from langchain_core.tools import tool
from app.services.index_service import IndexService

logger = logging.getLogger(__name__)

# 全局服务实例（单例，避免重复初始化 CLIP/Qdrant）
_index_service: IndexService | None = None


def _get_index_service() -> IndexService:
    global _index_service
    if _index_service is None:
        _index_service = IndexService()
    return _index_service


@tool
def search_materials(description: str, top_k: int = 10) -> str:
    """用自然语言描述搜索本地素材库，返回最匹配的素材列表。

    Args:
        description: 画面描述，如"飞机起飞升空，云层掠过"
        top_k: 返回结果数量，默认 10

    Returns:
        JSON 字符串：匹配素材列表 [{file_name, file_path, score, ...}]
    """
    service = _get_index_service()
    results = service.search_by_text(description, top_k=top_k)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def generate_material(description: str) -> str:
    """为某个镜头生成 AI 创作素材的优化 Prompt。

    Args:
        description: 镜头画面描述

    Returns:
        扩写后的 AI 生成 Prompt（中文）
    """
    enhanced = (
        f"电影感画面，高画质，{description}。"
        f"注重构图、光影和色彩，画面主体清晰，背景虚化，8K分辨率。"
    )
    return enhanced
