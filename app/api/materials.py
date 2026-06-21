from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project
from app.models.material import MaterialMatch
from app.agents.material import MaterialAgent
from app.api.deps import get_material_agent

router = APIRouter(prefix="/api/v1/projects", tags=["materials"])


class MaterialRequest(BaseModel):
    mode: str = "retrieval"  # "retrieval" | "generation"
    storyboard: dict
    material_overrides: dict | None = None


class CandidateResponse(BaseModel):
    material_id: str
    file_name: str
    clip_range: list[float]
    score: float
    reason: str


class MaterialResponse(BaseModel):
    shot_index: int
    candidates: list[CandidateResponse]
    suggestion: str | None = None


@router.post("/{project_id}/materials", response_model=list[MaterialResponse])
async def match_materials(
    project_id: str,
    req: MaterialRequest,
    db: AsyncSession = Depends(get_db),
    agent: MaterialAgent = Depends(get_material_agent),
):
    """素材Agent：检索匹配素材"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="项目不存在")

    if req.mode not in ("retrieval", "generation"):
        raise HTTPException(status_code=400, detail="mode 必须为 retrieval 或 generation")

    matches = await agent.match_materials(
        storyboard=req.storyboard,
        mode=req.mode,
        material_overrides=req.material_overrides,
    )

    return [
        MaterialResponse(
            shot_index=m["shot_index"],
            candidates=[
                CandidateResponse(
                    material_id=c["material_id"],
                    file_name=c["file_name"],
                    clip_range=c["clip_range"],
                    score=c["score"],
                    reason=c.get("reason", ""),
                )
                for c in m.get("candidates", [])
            ],
            suggestion=m.get("suggestion"),
        )
        for m in matches
    ]
