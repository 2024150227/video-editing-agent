from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project
from app.models.storyboard import Storyboard, Shot
from app.agents.director import DirectorAgent
from app.api.deps import get_director_agent

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
    storyboard_id: str
    project_id: str
    style: str
    total_duration_sec: int
    bgm_mood: str | None = None
    shots: list[ShotResponse]


@router.post("/{project_id}/story", response_model=StoryboardResponse)
async def generate_storyboard(
    project_id: str,
    req: GenerateStoryRequest,
    db: AsyncSession = Depends(get_db),
    director: DirectorAgent = Depends(get_director_agent),
):
    """导演Agent：生成分镜脚本"""
    # 校验项目存在
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    # 调用导演 Agent
    storyboard_dict = await director.generate_storyboard(
        prompt=req.prompt,
        style=req.style,
        total_duration_sec=req.total_duration_sec,
    )

    if "raw" in storyboard_dict:
        raise HTTPException(status_code=422, detail=f"LLM 返回格式异常: {storyboard_dict['error']}")

    # 持久化
    storyboard = Storyboard(
        project_id=project_id,
        style=storyboard_dict["style"],
        total_duration_sec=storyboard_dict["total_duration_sec"],
        bgm_mood=storyboard_dict.get("bgm_mood", ""),
        raw_prompt=req.prompt,
    )
    db.add(storyboard)
    await db.flush()

    for shot_data in storyboard_dict["shots"]:
        shot = Shot(
            storyboard_id=storyboard.id,
            index=shot_data["index"],
            duration_sec=shot_data["duration_sec"],
            description=shot_data["description"],
            shot_type=shot_data["shot_type"],
            camera_motion=shot_data.get("camera_motion", "静态"),
            transition=shot_data.get("transition", "硬切"),
            mood_words=shot_data.get("mood_words", []),
        )
        db.add(shot)

    await db.commit()
    await db.refresh(storyboard)

    shots = [
        ShotResponse(
            index=s.index,
            duration_sec=s.duration_sec,
            description=s.description,
            shot_type=s.shot_type,
            camera_motion=s.camera_motion,
            transition=s.transition,
            mood_words=s.mood_words,
        )
        for s in storyboard.shots
    ]

    return StoryboardResponse(
        storyboard_id=storyboard.id,
        project_id=project_id,
        style=storyboard.style,
        total_duration_sec=storyboard.total_duration_sec,
        bgm_mood=storyboard.bgm_mood,
        shots=shots,
    )
