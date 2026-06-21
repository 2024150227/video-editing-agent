from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project
from app.models.task import EditTask, TaskStatus
from app.agents.editor import EditorAgent
from app.api.deps import get_editor_agent

router = APIRouter(prefix="/api/v1/projects", tags=["edit"])


class EditRequest(BaseModel):
    storyboard: dict
    material_selections: list[dict]
    bgm_path: str | None = None
    subtitles: str | None = None


class EditResponse(BaseModel):
    task_id: str
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
    editor: EditorAgent = Depends(get_editor_agent),
):
    """剪辑Agent：提交渲染任务"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="项目不存在")

    # 并发控制：检查是否有进行中的任务
    existing = await db.execute(
        select(EditTask).where(
            EditTask.project_id == project_id,
            EditTask.status.in_([TaskStatus.QUEUED, TaskStatus.PROCESSING, TaskStatus.RENDERING]),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=429, detail="该项目已有进行中的渲染任务")

    # 提交异步任务
    task_id = editor.submit_edit(
        project_id=project_id,
        storyboard=req.storyboard,
        material_selections=req.material_selections,
        bgm_path=req.bgm_path,
    )

    # 持久化任务记录
    edit_task = EditTask(
        id=task_id,
        project_id=project_id,
        storyboard_id=req.storyboard.get("storyboard_id", ""),
        status=TaskStatus.QUEUED,
    )
    db.add(edit_task)
    await db.commit()

    return EditResponse(task_id=task_id, status="queued")


@router.get("/{project_id}/status", response_model=TaskStatusResponse)
async def get_edit_status(
    project_id: str,
    task_id: str,
    db: AsyncSession = Depends(get_db),
    editor: EditorAgent = Depends(get_editor_agent),
):
    """查询渲染任务状态"""
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
async def get_edit_result(
    project_id: str,
    task_id: str,
    editor: EditorAgent = Depends(get_editor_agent),
):
    """获取最终视频"""
    result = editor.get_output_url(task_id)
    return TaskResultResponse(
        ready=result["ready"],
        file_path=result.get("file_path"),
        file_size_bytes=result.get("file_size_bytes"),
        error=result.get("error"),
    )
