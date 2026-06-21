import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Integer, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
import enum


class TaskStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class EditTask(Base):
    __tablename__ = "edit_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    storyboard_id: Mapped[str] = mapped_column(String(36), ForeignKey("storyboards.id"), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(SAEnum(TaskStatus), default=TaskStatus.QUEUED)
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    estimated_remaining_sec: Mapped[int] = mapped_column(Integer, nullable=True)
    output_path: Mapped[str] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[str] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
