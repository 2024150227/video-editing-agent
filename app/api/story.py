"""导演 API — 启动 LangGraph，在分镜生成后中断返回"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project
from app.models.storyboard import Storyboard, Shot
from app.api.deps import get_graph

router = APIRouter(prefix="/api/v1/projects", tags=["story"])


class GenerateStoryRequest(BaseModel):
    prompt: str
    style: str = "默认风格"
    total_duration_sec: int = 30


class ShotResponse(BaseModel):
    index: int
    duration_sec: int
    description: str
    shot_type: str
    camera_motion: str = "静态"
    transition: str = "硬切"
    mood_words: list[str] = []


class StoryboardResponse(BaseModel):
    storyboard_id: str | None = None
    project_id: str
    style: str
    total_duration_sec: int
    bgm_mood: str | None = None
    shots: list[ShotResponse]
    step: str = "storyboard_review"
    message: str = "请审核分镜脚本，修改后通过 /materials 端点继续"


@router.post("/{project_id}/story", response_model=StoryboardResponse)
async def generate_storyboard(
    project_id: str,
    req: GenerateStoryRequest,
    db: AsyncSession = Depends(get_db),
    graph=Depends(get_graph),
):
    """启动 LangGraph，生成分镜后在中断点返回"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    # 构建初始状态
    initial_state = {
        "project_id": project_id,
        "prompt": req.prompt,
        "style": req.style,
        "total_duration_sec": req.total_duration_sec,
        "mode": "retrieval",
        "bgm_path": None,
        "storyboard": None,
        "material_matches": None,
        "material_overrides": None,
        "material_selections": None,
        "task_id": None,
        "task_status": None,
        "messages": [],
        "current_step": "init",
        "error": None,
    }

    config = {"configurable": {"thread_id": project_id}}

    # graph.invoke() 运行到第一个 interrupt（在 director_node 中）
    final_state = await graph.ainvoke(initial_state, config)

    storyboard = final_state.get("storyboard") or {}

    # 持久化分镜
    if "raw" not in storyboard and storyboard.get("shots"):
        storyboard_orm = Storyboard(
            project_id=project_id,
            style=storyboard.get("style", ""),
            total_duration_sec=storyboard.get("total_duration_sec", req.total_duration_sec),
            bgm_mood=storyboard.get("bgm_mood", ""),
            raw_prompt=req.prompt,
        )
        db.add(storyboard_orm)
        await db.flush()

        for shot_data in storyboard.get("shots", []):
            shot = Shot(
                storyboard_id=storyboard_orm.id,
                index=shot_data.get("index", 0),
                duration_sec=shot_data.get("duration_sec", 3),
                description=shot_data.get("description", ""),
                shot_type=shot_data.get("shot_type", "wide"),
                camera_motion=shot_data.get("camera_motion", "静态"),
                transition=shot_data.get("transition", "硬切"),
                mood_words=shot_data.get("mood_words", []),
            )
            db.add(shot)

        await db.commit()
        storyboard_id = storyboard_orm.id
    else:
        storyboard_id = None

    return StoryboardResponse(
        storyboard_id=storyboard_id,
        project_id=project_id,
        style=storyboard.get("style", ""),
        total_duration_sec=storyboard.get("total_duration_sec", req.total_duration_sec),
        bgm_mood=storyboard.get("bgm_mood"),
        shots=[
            ShotResponse(
                index=s.get("index", 0),
                duration_sec=s.get("duration_sec", 3),
                description=s.get("description", ""),
                shot_type=s.get("shot_type", "wide"),
                camera_motion=s.get("camera_motion", "静态"),
                transition=s.get("transition", "硬切"),
                mood_words=s.get("mood_words", []),
            )
            for s in storyboard.get("shots", [])
        ],
        step="storyboard_review",
    )
