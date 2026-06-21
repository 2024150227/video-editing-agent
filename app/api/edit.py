"""剪辑 API — 确认素材后继续 LangGraph 到剪辑完成，以及状态/结果查询"""

from langgraph.types import Command
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project
from app.models.task import EditTask, TaskStatus
from app.agents.editor import EditorAgent
from app.api.deps import get_graph

router = APIRouter(prefix="/api/v1/projects", tags=["edit"])


class EditRequest(BaseModel):
    storyboard: dict
    material_selections: list[dict]
    bgm_path: str | None = None
    subtitles: str | None = None


class EditResponse(BaseModel):
    task_id: str | None = None
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress_pct: int
    estimated_remaining_sec: int | None = None
    output_path: str | None = None
    error: str | None = None


class TaskResultResponse(BaseModel):
    ready: bool
    file_path: str | None = None
    file_size_bytes: int | None = None
    error: str | None = None


@router.post("/{project_id}/edit", response_model=EditResponse)
async def start_edit(
    project_id: str,
    req: EditRequest,
    db: AsyncSession = Depends(get_db),
    graph=Depends(get_graph),
):
    """继续 LangGraph — 素材确认后执行剪辑节点，同步等待渲染完成"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="项目不存在")

    # 并发控制
    existing = await db.execute(
        select(EditTask).where(
            EditTask.project_id == project_id,
            EditTask.status.in_([TaskStatus.QUEUED, TaskStatus.PROCESSING, TaskStatus.RENDERING]),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=429, detail="该项目已有进行中的渲染任务")

    config = {"configurable": {"thread_id": project_id}}

    # resume 传递用户确认的分镜和素材选择
    resume_value = {
        "storyboard": req.storyboard,
    }

    # 继续执行 graph — 从 material_node 中断后运行到 editor_node → END
    final_state = await graph.ainvoke(
        Command(resume=resume_value),
        config,
    )

    task_id = final_state.get("task_id")
    task_status = final_state.get("task_status") or {}

    return EditResponse(
        task_id=task_id,
        status=task_status.get("status", "unknown"),
    )


@router.get("/{project_id}/status", response_model=TaskStatusResponse)
async def get_edit_status(project_id: str, task_id: str):
    """查询渲染任务状态"""
    editor = EditorAgent()
    status = editor.get_task_status(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        status=status["status"],
        progress_pct=status["progress_pct"],
        estimated_remaining_sec=status.get("estimated_remaining_sec"),
        output_path=status.get("output_path"),
        error=status.get("error"),
    )


@router.get("/{project_id}/result", response_model=TaskResultResponse)
async def get_edit_result(project_id: str, task_id: str):
    """获取最终视频"""
    editor = EditorAgent()
    result = editor.get_output_url(task_id)
    return TaskResultResponse(
        ready=result["ready"],
        file_path=result.get("file_path"),
        file_size_bytes=result.get("file_size_bytes"),
        error=result.get("error"),
    )
