"""Video Editing Agent — LangGraph State 定义"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class VideoEditingState(TypedDict):
    """全程流转的状态对象，所有节点通过它共享数据"""

    # ── 用户输入 ──
    project_id: str
    prompt: str
    style: str
    total_duration_sec: int
    mode: str                       # "retrieval" | "generation"
    bgm_path: str | None

    # ── 导演节点输出 ──
    storyboard: dict | None         # 分镜脚本

    # ── 素材节点输出 ──
    material_matches: list[dict] | None
    material_overrides: dict | None
    material_selections: list[dict] | None

    # ── 剪辑节点输出 ──
    task_id: str | None
    task_status: dict | None

    # ── 对话消息（LangGraph 内置 add_messages reducer）──
    messages: Annotated[list, add_messages]

    # ── 流程控制 ──
    current_step: str               # "init" | "storyboard_review" | "material_review" | "rendering" | "done"
    error: str | None
