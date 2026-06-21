import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Integer, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Storyboard(Base):
    __tablename__ = "storyboards"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    style: Mapped[str] = mapped_column(String(100), nullable=False)
    total_duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    bgm_mood: Mapped[str] = mapped_column(String(200), nullable=True)
    raw_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    shots: Mapped[list["Shot"]] = relationship("Shot", back_populates="storyboard", cascade="all, delete-orphan")
    matches: Mapped[list["MaterialMatch"]] = relationship("MaterialMatch", back_populates="storyboard")


class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    storyboard_id: Mapped[str] = mapped_column(String(36), ForeignKey("storyboards.id"), nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    shot_type: Mapped[str] = mapped_column(String(50), nullable=False)
    camera_motion: Mapped[str] = mapped_column(String(50), default="静态")
    transition: Mapped[str] = mapped_column(String(50), default="硬切")
    mood_words: Mapped[list] = mapped_column(JSON, default=lambda: [])

    storyboard: Mapped["Storyboard"] = relationship("Storyboard", back_populates="shots")
