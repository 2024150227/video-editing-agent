"""LangGraph Builder — 构建视频剪辑 Agent 的状态图"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.graph.state import VideoEditingState
from app.core.graph.nodes import director_node, material_node, editor_node


def build_graph() -> StateGraph:
    """构建并编译 LangGraph 状态图

    Returns:
        编译后的 StateGraph，可直接调用 .invoke() / .astream()
    """

    # 创建状态图
    workflow = StateGraph(VideoEditingState)

    # 注册三个核心节点
    workflow.add_node("director", director_node)
    workflow.add_node("material", material_node)
    workflow.add_node("editor", editor_node)

    # 定义流程：START → director → material → editor → END
    workflow.add_edge(START, "director")
    workflow.add_edge("director", "material")
    workflow.add_edge("material", "editor")
    workflow.add_edge("editor", END)

    # 编译，带 MemorySaver 支持 checkpoint / interrupt / resume
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph
