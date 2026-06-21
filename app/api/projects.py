from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project, IndexStatus
import uuid

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    name: str
    material_source_dir: str
    aspect_ratio: str = "9:16"
    resolution: str = "1080x1920"


class ProjectResponse(BaseModel):
    project_id: str
    name: str
    material_source_dir: str
    index_status: str
    created_at: str


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(req: CreateProjectRequest, db: AsyncSession = Depends(get_db)):
    """创建剪辑项目"""
    import os
    if not os.path.isdir(req.material_source_dir):
        raise HTTPException(status_code=400, detail=f"素材目录不存在: {req.material_source_dir}")

    project = Project(
        id=str(uuid.uuid4()),
        name=req.name,
        material_source_dir=req.material_source_dir,
        aspect_ratio=req.aspect_ratio,
        resolution=req.resolution,
        index_status=IndexStatus.PENDING,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    return ProjectResponse(
        project_id=project.id,
        name=project.name,
        material_source_dir=project.material_source_dir,
        index_status=project.index_status.value,
        created_at=project.created_at.isoformat(),
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db)):
    """获取项目信息"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    return ProjectResponse(
        project_id=project.id,
        name=project.name,
        material_source_dir=project.material_source_dir,
        index_status=project.index_status.value,
        created_at=project.created_at.isoformat(),
    )


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db)):
    """删除项目"""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    await db.delete(project)
    await db.commit()
