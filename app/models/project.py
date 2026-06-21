import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base

import enum


class IndexStatus(str, enum.Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    material_source_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    aspect_ratio: Mapped[str] = mapped_column(String(10), default="9:16")
    resolution: Mapped[str] = mapped_column(String(20), default="1080x1920")
    index_status: Mapped[IndexStatus] = mapped_column(
        SAEnum(IndexStatus), default=IndexStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
