"""素材 API — 确认分镜后继续 LangGraph，在素材匹配后中断返回"""

from langgraph.types import Command
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project
from app.api.deps import get_graph

router = APIRouter(prefix="/api/v1/projects", tags=["materials"])


class MaterialRequest(BaseModel):
    mode: str = "retrieval"          # "retrieval" | "generation"
    storyboard: dict | None = None   # 用户确认/修改后的分镜
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
    graph=Depends(get_graph),
):
    """继续 LangGraph — 素材节点执行，匹配完成后中断返回"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="项目不存在")

    if req.mode not in ("retrieval", "generation"):
        raise HTTPException(status_code=400, detail="mode 必须为 retrieval 或 generation")

    config = {"configurable": {"thread_id": project_id}}

    # 用户确认的分镜（可能修改过）作为 resume 值
    resume_value = {
        "storyboard": req.storyboard,
    }

    # 继续执行 graph，从 director_node 的 interrupt 之后继续
    final_state = await graph.ainvoke(
        Command(resume=resume_value),
        config,
    )

    # 提取素材匹配结果（在 interrupt 中或 state 中）
    matches = final_state.get("material_matches") or []
    if not matches:
        interrupts = final_state.get("__interrupt__", [])
        if interrupts:
            interrupt_data = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            if isinstance(interrupt_data, dict):
                matches = interrupt_data.get("material_matches", []) or []

    return [
        MaterialResponse(
            shot_index=m.get("shot_index", 0),
            candidates=[
                CandidateResponse(
                    material_id=c.get("material_id", ""),
                    file_name=c.get("file_name", ""),
                    clip_range=c.get("clip_range", [0, 3]),
                    score=c.get("score", 0.0),
                    reason=c.get("reason", ""),
                )
                for c in m.get("candidates", [])
            ],
            suggestion=m.get("suggestion"),
        )
        for m in matches
    ]
