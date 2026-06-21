import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Material(Base):
    __tablename__ = "materials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)  # video / image / audio
    duration_sec: Mapped[float] = mapped_column(Float, nullable=True)
    qdrant_point_id: Mapped[str] = mapped_column(String(36), nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MaterialMatch(Base):
    __tablename__ = "material_matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    storyboard_id: Mapped[str] = mapped_column(String(36), ForeignKey("storyboards.id"), nullable=False)
    shot_index: Mapped[int] = mapped_column(Integer, nullable=False)
    material_id: Mapped[str] = mapped_column(String(36), ForeignKey("materials.id"), nullable=False)
    clip_start_sec: Mapped[float] = mapped_column(Float, default=0.0)
    clip_end_sec: Mapped[float] = mapped_column(Float, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    user_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
