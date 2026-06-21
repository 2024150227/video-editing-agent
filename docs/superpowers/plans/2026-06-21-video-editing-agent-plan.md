# 智能视频剪辑 Agent — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于多模态大模型的智能视频剪辑 API 服务，支持文本→分镜→素材匹配→视频合成三步流程。

**Architecture:** FastAPI 提供 REST API，三个 Agent（导演/素材/剪辑）各司其职，火山方舟 LLM 做创意理解，Qdrant + CLIP 做跨模态检索，MoviePy + FFmpeg 做视频合成，Celery 处理异步渲染。

**Tech Stack:** Python 3.11+, FastAPI ≥0.110, MoviePy ≥2.0, Celery ≥5.3, Qdrant, CLIP, SQLAlchemy ≥2.0, PostgreSQL, Redis, FFmpeg, librosa, Pillow, Uvicorn

## Global Constraints

- Python 3.11+
- fastapi ≥0.110, moviepy ≥2.0, celery ≥5.3, qdrant-client ≥1.9, sqlalchemy ≥2.0, alembic ≥1.13
- 火山方舟 API 作为唯一多模态模型后端
- 单用户同时最多 1 个渲染任务
- LLM JSON 解析失败自动 retry × 3
- 所有视频输出默认 1080×1920 (9:16 竖屏)

---

### Task 1: 项目脚手架 & 依赖

**Files:**
- Create: `requirements.txt`
- Create: `docker-compose.yml`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/core/__init__.py`
- Create: `app/core/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `app.main:app` (FastAPI app instance), `app.core.config.Settings` (pydantic-settings), `docker-compose.yml` (postgres+redis+qdrant)

- [ ] **Step 1: 编写 requirements.txt**

```
fastapi>=0.110
uvicorn[standard]>=0.27
sqlalchemy>=2.0
alembic>=1.13
asyncpg>=0.29
psycopg2-binary>=2.9
pydantic>=2.5
pydantic-settings>=2.1
redis>=5.0
celery>=5.3
moviepy>=2.0
qdrant-client>=1.9
openai-clip>=1.0
pillow>=10.0
librosa>=0.10
ffmpeg-python>=0.2
httpx>=0.25
python-multipart>=0.0.6
pytest>=8.0
pytest-asyncio>=0.23
pytest-mock>=3.12
```

- [ ] **Step 2: 编写 docker-compose.yml**

```yaml
version: "3.9"

services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: vagent
      POSTGRES_PASSWORD: vagent_secret
      POSTGRES_DB: vagent
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  pgdata:
  qdrant_data:
```

- [ ] **Step 3: 编写 app/core/config.py**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 应用
    APP_NAME: str = "Video Editing Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # 数据库
    DATABASE_URL: str = "postgresql+asyncpg://vagent:vagent_secret@localhost:5432/vagent"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://vagent:vagent_secret@localhost:5432/vagent"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "materials"

    # 火山方舟
    VOLCANO_ARK_API_KEY: str = ""
    VOLCANO_ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    VOLCANO_ARK_MODEL: str = "doubao-vision-pro-32k"

    # CLIP
    CLIP_MODEL: str = "ViT-B/32"

    # 视频
    DEFAULT_RESOLUTION: str = "1080x1920"
    DEFAULT_ASPECT_RATIO: str = "9:16"
    DEFAULT_FPS: int = 30
    OUTPUT_DIR: str = "./output"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # 并发控制
    MAX_CONCURRENT_RENDERS: int = 1

    # 中文字体路径 (用于字幕)
    FONT_PATH: str = "/usr/share/fonts/NotoSansSC-Regular.ttf"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: 编写 app/main.py**

```python
from fastapi import FastAPI
from app.core.config import get_settings

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()
```

- [ ] **Step 5: 编写 tests/conftest.py**

```python
import pytest
from fastapi.testclient import TestClient
from app.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def settings_override():
    from app.core.config import Settings
    return Settings(
        VOLCANO_ARK_API_KEY="test_key",
        DATABASE_URL="sqlite+aiosqlite:///:memory:",
        DATABASE_URL_SYNC="sqlite:///:memory:",
        OUTPUT_DIR="./test_output",
    )
```

- [ ] **Step 6: 验证项目能启动**

```bash
cd "E:/Desktop/基于多模态大模型的智能视频剪辑agent"
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
curl http://localhost:8000/health
# Expected: {"status":"ok","version":"0.1.0"}
kill %1
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt docker-compose.yml app/ tests/
git commit -m "feat: add project scaffold, config, and docker-compose"
```

---

### Task 2: 数据库模型 & 迁移

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/project.py`
- Create: `app/models/storyboard.py`
- Create: `app/models/material.py`
- Create: `app/models/task.py`
- Create: `app/core/database.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/.gitkeep`

**Interfaces:**
- Produces: `app.core.database.get_db()` (async generator yielding AsyncSession), ORM models: `Project`, `Storyboard`, `Shot`, `Material`, `EditTask`

- [ ] **Step 1: 编写 app/core/database.py**

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
```

- [ ] **Step 2: 编写 app/models/project.py**

```python
import uuid
from datetime import datetime
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

    id: Mapped[str] = mapped_column(String(36), primary_key, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    material_source_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    aspect_ratio: Mapped[str] = mapped_column(String(10), default="9:16")
    resolution: Mapped[str] = mapped_column(String(20), default="1080x1920")
    index_status: Mapped[IndexStatus] = mapped_column(
        SAEnum(IndexStatus), default=IndexStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

- [ ] **Step 3: 编写 app/models/storyboard.py**

```python
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Integer, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Storyboard(Base):
    __tablename__ = "storyboards"

    id: Mapped[str] = mapped_column(String(36), primary_key, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    style: Mapped[str] = mapped_column(String(100), nullable=False)
    total_duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    bgm_mood: Mapped[str] = mapped_column(String(200), nullable=True)
    raw_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    shots: Mapped[list["Shot"]] = relationship("Shot", back_populates="storyboard", cascade="all, delete-orphan")


class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[str] = mapped_column(String(36), primary_key, default=lambda: str(uuid.uuid4()))
    storyboard_id: Mapped[str] = mapped_column(String(36), ForeignKey("storyboards.id"), nullable=False)
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    shot_type: Mapped[str] = mapped_column(String(50), nullable=False)
    camera_motion: Mapped[str] = mapped_column(String(50), default="静态")
    transition: Mapped[str] = mapped_column(String(50), default="硬切")
    mood_words: Mapped[dict] = mapped_column(JSON, default=list)

    storyboard: Mapped["Storyboard"] = relationship("Storyboard", back_populates="shots")
```

- [ ] **Step 4: 编写 app/models/material.py**

```python
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Material(Base):
    __tablename__ = "materials"

    id: Mapped[str] = mapped_column(String(36), primary_key, default=lambda: str(uuid.uuid4()))
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

    id: Mapped[str] = mapped_column(String(36), primary_key, default=lambda: str(uuid.uuid4()))
    storyboard_id: Mapped[str] = mapped_column(String(36), ForeignKey("storyboards.id"), nullable=False)
    shot_index: Mapped[int] = mapped_column(Integer, nullable=False)
    material_id: Mapped[str] = mapped_column(String(36), ForeignKey("materials.id"), nullable=False)
    clip_start_sec: Mapped[float] = mapped_column(Float, default=0.0)
    clip_end_sec: Mapped[float] = mapped_column(Float, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    user_selected: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 5: 编写 app/models/task.py**

```python
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, ForeignKey, Enum as SAEnum
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

    id: Mapped[str] = mapped_column(String(36), primary_key, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id"), nullable=False)
    storyboard_id: Mapped[str] = mapped_column(String(36), ForeignKey("storyboards.id"), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(SAEnum(TaskStatus), default=TaskStatus.QUEUED)
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    estimated_remaining_sec: Mapped[int] = mapped_column(Integer, nullable=True)
    output_path: Mapped[str] = mapped_column(String(1024), nullable=True)
    error_message: Mapped[str] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

- [ ] **Step 6: 编写 app/models/__init__.py**

```python
from app.models.project import Project, IndexStatus
from app.models.storyboard import Storyboard, Shot
from app.models.material import Material, MaterialMatch
from app.models.task import EditTask, TaskStatus
from app.core.database import Base

__all__ = [
    "Base", "Project", "IndexStatus",
    "Storyboard", "Shot",
    "Material", "MaterialMatch",
    "EditTask", "TaskStatus",
]
```

- [ ] **Step 7: 配置 alembic & 生成初始迁移**

```bash
cd "E:/Desktop/基于多模态大模型的智能视频剪辑agent"
alembic init alembic
# 修改 alembic/env.py 使用 app.core.database.Base.metadata
# 修改 alembic.ini 中 sqlalchemy.url 指向 DATABASE_URL_SYNC
alembic revision --autogenerate -m "initial schema"
```

- [ ] **Step 8: Commit**

```bash
git add app/models/ app/core/database.py alembic.ini alembic/
git commit -m "feat: add database models and alembic migrations"
```

---

### Task 3: 火山方舟 LLM 客户端

**Files:**
- Create: `app/core/llm_client.py`
- Create: `tests/test_llm_client.py`

**Interfaces:**
- Produces: `LLMClient.chat(messages, temperature, max_tokens) -> str`, `LLMClient.chat_json(messages, temperature, max_tokens, retries) -> dict`

- [ ] **Step 1: 编写测试 tests/test_llm_client.py**

```python
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.llm_client import LLMClient


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_chat_returns_text_on_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "你好，世界"}}]
        }

        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat([{"role": "user", "content": "你好"}])
            assert result == "你好，世界"

    @pytest.mark.asyncio
    async def test_chat_json_parses_valid_json(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"key": "value"}'}}]
        }

        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat_json([{"role": "user", "content": "返回JSON"}])
            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_chat_json_retries_on_invalid_json(self):
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.status_code = 200
            if call_count < 3:
                mock.json.return_value = {"choices": [{"message": {"content": "not json {"}}]}
            else:
                mock.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
            return mock

        with patch.object(httpx.AsyncClient, "post", side_effect=side_effect):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat_json([{"role": "user", "content": "JSON please"}], retries=3)
            assert result == {"ok": True}
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_chat_json_returns_raw_on_exhausted_retries(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "totally broken"}}]}

        with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response)):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.chat_json([{"role": "user", "content": "JSON please"}], retries=2)
            assert result == {"raw": "totally broken", "error": "JSON parse failed after 2 retries"}
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_llm_client.py -v
# Expected: FAIL (module not found)
```

- [ ] **Step 3: 编写 app/core/llm_client.py**

```python
import json
import logging
import httpx
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """火山方舟 LLM API 客户端"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.VOLCANO_ARK_API_KEY
        self.base_url = base_url or settings.VOLCANO_ARK_BASE_URL
        self.model = model or settings.VOLCANO_ARK_MODEL

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """发送对话请求，返回文本响应"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> dict:
        """发送请求并强制解析 JSON 返回，解析失败自动重试"""
        last_raw = ""
        for attempt in range(retries):
            raw = await self.chat(messages, temperature, max_tokens)
            last_raw = raw
            try:
                # 尝试提取 JSON 块
                text = raw.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return json.loads(text)
            except (json.JSONDecodeError, IndexError):
                logger.warning(f"JSON parse failed, attempt {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    messages.append({"role": "user", "content": "请严格按照 JSON 格式返回，不要包含其他文字。"})

        return {"raw": last_raw, "error": f"JSON parse failed after {retries} retries"}
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_llm_client.py -v
# Expected: ALL PASS
```

- [ ] **Step 5: Commit**

```bash
git add app/core/llm_client.py tests/test_llm_client.py
git commit -m "feat: add Volcano Ark LLM client with JSON mode retry"
```

---

### Task 4: CLIP Embedding 服务

**Files:**
- Create: `app/core/embedding.py`
- Create: `tests/test_embedding.py`

**Interfaces:**
- Produces: `EmbeddingService.encode_text(text) -> list[float]`, `EmbeddingService.encode_image(image_path) -> list[float]`

- [ ] **Step 1: 编写测试 tests/test_embedding.py**

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.core.embedding import EmbeddingService


class TestEmbeddingService:
    def test_encode_text_returns_512d_vector(self):
        mock_clip = MagicMock()
        mock_clip.encode_text.return_value = np.random.randn(1, 512).astype(np.float32)

        with patch("app.core.embedding.clip", MagicMock()):
            with patch("app.core.embedding.clip.load", return_value=(mock_clip, None)):
                service = EmbeddingService()
                result = service.encode_text("一只猫在沙发上")
                assert isinstance(result, list)
                assert len(result) == 512

    def test_encode_image_returns_512d_vector(self):
        mock_clip = MagicMock()
        mock_clip.encode_image.return_value = np.random.randn(1, 512).astype(np.float32)

        with patch("app.core.embedding.clip", MagicMock()):
            with patch("app.core.embedding.clip.load", return_value=(mock_clip, None)):
                service = EmbeddingService()
                result = service.encode_image("/fake/path.jpg")
                assert isinstance(result, list)
                assert len(result) == 512

    def test_encode_text_is_deterministic(self):
        mock_clip = MagicMock()
        vec = np.random.randn(1, 512).astype(np.float32)
        mock_clip.encode_text.return_value = vec

        with patch("app.core.embedding.clip", MagicMock()):
            with patch("app.core.embedding.clip.load", return_value=(mock_clip, None)):
                service = EmbeddingService()
                r1 = service.encode_text("test")
                r2 = service.encode_text("test")
                assert r1 == r2
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_embedding.py -v
# Expected: FAIL
```

- [ ] **Step 3: 编写 app/core/embedding.py**

```python
import logging
import numpy as np
import clip
import torch
from PIL import Image
from functools import lru_cache
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """CLIP 跨模态 Embedding 服务 — 文本和图像映射到同一向量空间"""

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        settings = get_settings()
        self.model_name = model_name or settings.CLIP_MODEL
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = None, None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading CLIP model '{self.model_name}' on {self.device}...")
        self._model, self._preprocess = clip.load(self.model_name, device=self.device)
        self._model.eval()

    def encode_text(self, text: str) -> list[float]:
        """将文本编码为向量"""
        tokenized = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokenized)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()

    def encode_image(self, image_path: str) -> list[float]:
        """将图片编码为向量"""
        image = Image.open(image_path).convert("RGB")
        image_input = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0].tolist()
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_embedding.py -v
# Expected: ALL PASS
```

- [ ] **Step 5: Commit**

```bash
git add app/core/embedding.py tests/test_embedding.py
git commit -m "feat: add CLIP embedding service for text-image cross-modal encoding"
```

---

### Task 5: 导演 Agent — 分镜生成

**Files:**
- Create: `app/agents/__init__.py`
- Create: `app/agents/director.py`
- Create: `tests/test_director.py`

**Interfaces:**
- Consumes: `LLMClient` (from Task 3)
- Produces: `DirectorAgent.generate_storyboard(prompt, style, total_duration_sec) -> StoryboardDict`

- [ ] **Step 1: 编写测试 tests/test_director.py**

```python
import pytest
from unittest.mock import AsyncMock, patch
from app.agents.director import DirectorAgent


class TestDirectorAgent:
    @pytest.fixture
    def valid_storyboard_json(self):
        return {
            "style": "快节奏卡点",
            "total_duration_sec": 30,
            "bgm_mood": "电子/动感",
            "shots": [
                {
                    "index": 1,
                    "duration_sec": 3,
                    "description": "飞机起飞升空，云层掠过",
                    "shot_type": "wide",
                    "camera_motion": "静态",
                    "transition": "硬切",
                    "mood_words": ["自由", "启程"]
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_generate_storyboard_returns_valid_structure(self, valid_storyboard_json):
        with patch("app.agents.director.LLMClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.chat_json = AsyncMock(return_value=valid_storyboard_json)

            agent = DirectorAgent()
            result = await agent.generate_storyboard(
                prompt="旅行素材30秒快剪",
                style="快节奏",
                total_duration_sec=30,
            )
            assert result["style"] == "快节奏卡点"
            assert result["total_duration_sec"] == 30
            assert len(result["shots"]) > 0
            assert "description" in result["shots"][0]

    @pytest.mark.asyncio
    async def test_duration_mismatch_triggers_fix(self, valid_storyboard_json):
        valid_storyboard_json["shots"][0]["duration_sec"] = 10  # 只有10秒，要求30秒

        with patch("app.agents.director.LLMClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.chat_json = AsyncMock(side_effect=[
                valid_storyboard_json,  # 第一次返回时长不匹配
                {  # retry 后修正
                    "style": "快节奏卡点",
                    "total_duration_sec": 30,
                    "bgm_mood": "电子",
                    "shots": [
                        {"index": 1, "duration_sec": 3, "description": "镜头1", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 2, "duration_sec": 3, "description": "镜头2", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 3, "duration_sec": 3, "description": "镜头3", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 4, "duration_sec": 3, "description": "镜头4", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 5, "duration_sec": 3, "description": "镜头5", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 6, "duration_sec": 3, "description": "镜头6", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 7, "duration_sec": 3, "description": "镜头7", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 8, "duration_sec": 3, "description": "镜头8", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 9, "duration_sec": 3, "description": "镜头9", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                        {"index": 10, "duration_sec": 3, "description": "镜头10", "shot_type": "wide", "camera_motion": "静态", "transition": "硬切", "mood_words": []},
                    ]
                }
            ])
            agent = DirectorAgent()
            result = await agent.generate_storyboard("test", "快节奏", 30)
            assert sum(s["duration_sec"] for s in result["shots"]) == 30
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_director.py -v
# Expected: FAIL
```

- [ ] **Step 3: 编写 app/agents/director.py**

```python
import json
import logging
from app.core.llm_client import LLMClient

logger = logging.getLogger(__name__)

DIRECTOR_SYSTEM_PROMPT = """你是一个专业的视频导演。根据用户的创意描述，生成详细的分镜脚本。
你必须严格返回 JSON 格式，包含以下字段：
- style: 视频风格描述
- total_duration_sec: 总时长（秒）
- bgm_mood: BGM情绪描述
- shots: 分镜列表，每个分镜包含：
  - index: 镜头序号（从1开始）
  - duration_sec: 该镜头时长（秒）
  - description: 画面内容描述（一句话，用于素材检索）
  - shot_type: 景别（wide/medium/close-up/detail）
  - camera_motion: 运镜方式（静态/推/拉/摇/移/跟/升/降）
  - transition: 转场（硬切/淡入/淡出/叠化/闪白）
  - mood_words: 该镜头的情绪关键词列表

要求：
1. 所有镜头的 duration_sec 之和必须精确等于 total_duration_sec
2. 镜头描述要具体，包含主体、动作、场景、光线等信息
3. 节奏与风格匹配：快节奏视频镜头时长短（2-4秒），慢节奏镜头时长长（5-8秒）
4. 每个镜头描述独立，可直接用于素材检索"""


class DirectorAgent:
    """导演 Agent — 将用户创意描述转化为结构化分镜脚本"""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    async def generate_storyboard(
        self,
        prompt: str,
        style: str = "默认风格",
        total_duration_sec: int = 30,
    ) -> dict:
        """生成分镜脚本并校验"""

        messages = self._build_user_message(prompt, style, total_duration_sec)

        # Step 1: 生成分镜
        storyboard = await self.llm.chat_json(
            messages=[{"role": "system", "content": DIRECTOR_SYSTEM_PROMPT}, messages],
            temperature=0.8,
            max_tokens=4096,
            retries=3,
        )

        # Step 2: 规则校验
        storyboard = self._validate_and_fix(storyboard, total_duration_sec)

        return storyboard

    def _build_user_message(self, prompt: str, style: str, total_duration_sec: int) -> dict:
        return {
            "role": "user",
            "content": f"创意描述：{prompt}\n目标风格：{style}\n要求总时长：{total_duration_sec} 秒\n请生成分镜脚本。",
        }

    def _validate_and_fix(self, storyboard: dict, expected_duration: int) -> dict:
        """校验分镜脚本，并在出错时尝试修复"""
        if "raw" in storyboard:
            return storyboard  # LLM 解析已经失败

        shots = storyboard.get("shots", [])
        actual_duration = sum(s.get("duration_sec", 0) for s in shots)

        # 校验时长
        if abs(actual_duration - expected_duration) > 1:
            logger.warning(
                f"Duration mismatch: expected {expected_duration}s, got {actual_duration}s"
            )
            # 尝试调整最后一个镜头的时长来补齐
            if shots and actual_duration > 0:
                diff = expected_duration - actual_duration
                shots[-1]["duration_sec"] = max(1, shots[-1]["duration_sec"] + diff)
                storybook["shots"] = shots
                storybook["total_duration_sec"] = expected_duration
                logger.info(f"Fixed duration: adjusted last shot by {diff}s")

        # 校验每个镜头必填字段
        required_fields = ["index", "duration_sec", "description", "shot_type", "transition", "mood_words"]
        for shot in shots:
            for field in required_fields:
                if field not in shot:
                    shot[field] = [] if field == "mood_words" else ""

        storybook["total_duration_sec"] = expected_duration
        return storyboard
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_director.py -v
# Expected: ALL PASS
```

- [ ] **Step 5: Commit**

```bash
git add app/agents/__init__.py app/agents/director.py tests/test_director.py
git commit -m "feat: add Director Agent with storyboard generation and validation"
```

---

### Task 6: 素材索引服务

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/file_service.py`
- Create: `app/services/index_service.py`
- Create: `tests/test_index_service.py`

**Interfaces:**
- Consumes: `EmbeddingService` (from Task 4), Qdrant client
- Produces: `IndexService.index_directory(project_id, dir_path) -> int` (returns count), `IndexService.search(query_vector, top_k) -> list[dict]`

- [ ] **Step 1: 编写 app/services/file_service.py**

```python
import os
import logging

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}


class FileService:
    """素材文件扫描服务"""

    @staticmethod
    def scan_directory(dir_path: str) -> list[dict]:
        """递归扫描目录，返回所有支持的媒体文件"""
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")

        files = []
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                all_exts = SUPPORTED_VIDEO | SUPPORTED_IMAGE | SUPPORTED_AUDIO
                if ext in all_exts:
                    full_path = os.path.join(root, fname)
                    file_type = (
                        "video" if ext in SUPPORTED_VIDEO
                        else "image" if ext in SUPPORTED_IMAGE
                        else "audio"
                    )
                    files.append({
                        "file_name": fname,
                        "file_path": full_path,
                        "file_type": file_type,
                        "ext": ext,
                    })
        logger.info(f"Scanned {dir_path}: found {len(files)} media files")
        return files

    @staticmethod
    def get_video_duration(file_path: str) -> float:
        """获取视频时长（秒），依赖 FFmpeg"""
        try:
            import subprocess
            import json
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            info = json.loads(result.stdout)
            return float(info["format"].get("duration", 0))
        except Exception as e:
            logger.warning(f"Failed to get duration for {file_path}: {e}")
            return 0.0

    @staticmethod
    def extract_keyframe(file_path: str, output_path: str, at_second: float = 1.0) -> bool:
        """从视频中提取关键帧图片"""
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-ss", str(at_second), "-i", file_path,
                "-vframes", "1", "-q:v", "2", output_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            logger.warning(f"Failed to extract keyframe from {file_path}: {e}")
            return False
```

- [ ] **Step 2: 编写测试 tests/test_index_service.py**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.index_service import IndexService


class TestIndexService:
    @pytest.fixture
    def mock_embedding(self):
        with patch("app.services.index_service.EmbeddingService") as mock:
            service = MagicMock()
            service.encode_image.return_value = [0.1] * 512
            service.encode_text.return_value = [0.1] * 512
            mock.return_value = service
            yield mock

    @pytest.fixture
    def mock_qdrant(self):
        with patch("app.services.index_service.QdrantClient") as mock:
            client = MagicMock()
            client.collection_exists.return_value = False
            client.search.return_value = [
                MagicMock(id="point_1", score=0.95, payload={"file_path": "/a/b.mp4", "file_name": "b.mp4"}),
                MagicMock(id="point_2", score=0.87, payload={"file_path": "/a/c.mp4", "file_name": "c.mp4"}),
            ]
            mock.return_value = client
            yield mock

    def test_search_returns_ranked_results(self, mock_embedding, mock_qdrant):
        service = IndexService()
        results = service.search([0.1] * 512, top_k=5)
        assert len(results) == 2
        assert results[0]["score"] > results[1]["score"]
        assert "file_path" in results[0]

    def test_search_handles_empty_results(self, mock_embedding, mock_qdrant):
        mock_qdrant.return_value.search.return_value = []
        service = IndexService()
        results = service.search([0.1] * 512, top_k=5)
        assert results == []
```

- [ ] **Step 3: 运行测试验证失败**

```bash
pytest tests/test_index_service.py -v
# Expected: FAIL
```

- [ ] **Step 4: 编写 app/services/index_service.py**

```python
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import get_settings
from app.core.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class IndexService:
    """素材索引服务 — 管理 Qdrant 向量数据库中的素材"""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_client: QdrantClient | None = None,
    ):
        settings = get_settings()
        self.embedding = embedding_service or EmbeddingService()
        self.client = qdrant_client or QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        )
        self.collection = settings.QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    def index_single(self, project_id: str, file_info: dict, keyframe_path: str) -> str:
        """索引单个素材（图片或视频关键帧）"""
        vector = self.embedding.encode_image(keyframe_path)
        point_id = f"{project_id}_{file_info['file_name']}_{hash(file_info['file_path'])}"

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "project_id": project_id,
                        "file_name": file_info["file_name"],
                        "file_path": file_info["file_path"],
                        "file_type": file_info["file_type"],
                    },
                )
            ],
        )
        return point_id

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """向量相似度检索，返回 Top-K 结果"""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {
                "point_id": r.id,
                "score": round(r.score, 4),
                "file_path": r.payload.get("file_path", ""),
                "file_name": r.payload.get("file_name", ""),
                "file_type": r.payload.get("file_type", ""),
                "project_id": r.payload.get("project_id", ""),
            }
            for r in results
        ]

    def search_by_text(self, text: str, top_k: int = 10) -> list[dict]:
        """用文本描述搜索素材"""
        vector = self.embedding.encode_text(text)
        return self.search(vector, top_k)

    def delete_project(self, project_id: str):
        """删除项目关联的所有向量"""
        self.client.delete(
            collection_name=self.collection,
            points_selector={"filter": {"must": [{"key": "project_id", "match": {"value": project_id}}]}},
        )
```

- [ ] **Step 5: 运行测试验证通过**

```bash
pytest tests/test_index_service.py -v
# Expected: ALL PASS
```

- [ ] **Step 6: Commit**

```bash
git add app/services/__init__.py app/services/file_service.py app/services/index_service.py tests/test_index_service.py
git commit -m "feat: add file scanning and material indexing service with Qdrant"
```

---

### Task 7: 素材 Agent — 检索 & 生成

**Files:**
- Create: `app/agents/material.py`
- Create: `tests/test_material.py`

**Interfaces:**
- Consumes: `IndexService` (from Task 6), `LLMClient` (from Task 3)
- Produces: `MaterialAgent.match_materials(storyboard, mode, overrides) -> list[dict]`

- [ ] **Step 1: 编写测试 tests/test_material.py**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.material import MaterialAgent


class TestMaterialAgent:
    @pytest.fixture
    def storyboard(self):
        return {
            "shots": [
                {"index": 1, "description": "飞机起飞，云层掠过", "duration_sec": 3},
                {"index": 2, "description": "海滩日落，金色光线", "duration_sec": 2},
            ]
        }

    @pytest.fixture
    def mock_llm(self):
        with patch("app.agents.material.LLMClient") as mock:
            client = MagicMock()
            client.chat_json = AsyncMock(return_value={
                "shot_index": 1,
                "candidates": [
                    {
                        "material_id": "m_001",
                        "file_name": "sky.mp4",
                        "clip_range": [0.0, 3.0],
                        "score": 0.95,
                        "reason": "画面包含天空和云层"
                    }
                ]
            })
            mock.return_value = client
            yield mock

    def test_retrieval_mode_calls_index_service(self, storyboard, mock_llm):
        with patch("app.agents.material.IndexService") as MockIndex:
            mock_index = MagicMock()
            mock_index.search_by_text.return_value = [
                {"point_id": "p1", "score": 0.92, "file_path": "/data/sky.mp4", "file_name": "sky.mp4"}
            ]
            MockIndex.return_value = mock_index

            agent = MaterialAgent()
            results = agent.match_materials_retrieval(storyboard)

            assert len(results) == 2  # 两个 shot 各一组结果
            mock_index.search_by_text.assert_called()

    def test_generation_mode_returns_generated_urls(self, storyboard, mock_llm):
        agent = MaterialAgent()
        results = agent.match_materials_generation(storyboard)

        assert len(results) == 2  # 两个 shot 各一组结果
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_material.py -v
# Expected: FAIL
```

- [ ] **Step 3: 编写 app/agents/material.py**

```python
import logging
import uuid
from app.core.llm_client import LLMClient
from app.services.index_service import IndexService

logger = logging.getLogger(__name__)

MATERIAL_RANK_PROMPT = """你是一个视频素材匹配专家。根据分镜描述，从候选素材中选出最匹配的 3-5 个。
对每个候选，给出：
- material_id: 素材唯一标识
- file_name: 文件名
- clip_range: 建议裁剪片段 [start_sec, end_sec]，基于镜头时长
- score: 匹配度 0-1
- reason: 匹配理由（一句话中文）

返回 JSON 格式。"""

GENERATION_PROMPT_TEMPLATE = """你是一个视频创作助手。请根据以下镜头描述，扩写为适合 AI 图片/视频生成的 Prompt。
要求：
1. 补充构图方式、光影条件、色彩风格
2. 保持画面主体清晰
3. 风格：电影感、高画质
4. 仅返回扩写后的 Prompt，不要其他内容

镜头描述：{description}"""


class MaterialAgent:
    """素材 Agent — 跨模态检索 + AI 生成"""

    def __init__(self, llm_client: LLMClient | None = None, index_service: IndexService | None = None):
        self.llm = llm_client or LLMClient()
        self.index = index_service or IndexService()

    def match_materials_retrieval(
        self,
        storyboard: dict,
        material_overrides: dict | None = None,
    ) -> list[dict]:
        """检索模式：向量搜索 + LLM 精排"""
        overrides = material_overrides or {}
        results = []

        for shot in storyboard["shots"]:
            shot_index = shot["index"]

            # 用户指定了特定文件则跳过检索
            if str(shot_index) in overrides:
                results.append({
                    "shot_index": shot_index,
                    "candidates": [{
                        "material_id": str(uuid.uuid4()),
                        "file_name": overrides[str(shot_index)],
                        "clip_range": [0, shot["duration_sec"]],
                        "score": 1.0,
                        "reason": "用户指定素材",
                    }],
                })
                continue

            # 粗排：向量检索 Top-10
            candidates = self.index.search_by_text(shot["description"], top_k=10)

            if not candidates:
                results.append({
                    "shot_index": shot_index,
                    "candidates": [],
                    "suggestion": "未找到匹配素材，建议降低匹配阈值或切换到 AI 生成模式",
                })
                continue

            # 精排方案：当前直接使用粗排 Top-3（LLM 精排调用可后续接入）
            top_candidates = []
            for c in candidates[:3]:
                top_candidates.append({
                    "material_id": c["point_id"],
                    "file_name": c["file_name"],
                    "clip_range": [0.0, float(shot["duration_sec"])],
                    "score": round(c["score"], 4),
                    "reason": f"语义匹配得分 {c['score']:.2f}",
                })

            results.append({
                "shot_index": shot_index,
                "candidates": top_candidates,
            })

        return results

    def match_materials_generation(self, storyboard: dict) -> list[dict]:
        """生成模式：为每个镜头生成扩写 Prompt（实际调用生成 API 在后续任务接入）"""
        results = []

        for shot in storyboard["shots"]:
            enhanced_prompt = GENERATION_PROMPT_TEMPLATE.format(
                description=shot["description"]
            )
            results.append({
                "shot_index": shot["index"],
                "candidates": [{
                    "material_id": str(uuid.uuid4()),
                    "file_name": f"generated_shot_{shot['index']}.png",
                    "clip_range": [0, shot["duration_sec"]],
                    "score": 1.0,
                    "reason": "AI 生成素材",
                    "generation_prompt": enhanced_prompt,
                }],
            })

        return results

    async def match_materials(
        self,
        storyboard: dict,
        mode: str = "retrieval",
        material_overrides: dict | None = None,
    ) -> list[dict]:
        """统一入口：根据模式调用检索或生成"""
        if mode == "generation":
            return self.match_materials_generation(storyboard)
        return self.match_materials_retrieval(storyboard, material_overrides)
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_material.py -v
# Expected: ALL PASS
```

- [ ] **Step 5: Commit**

```bash
git add app/agents/material.py tests/test_material.py
git commit -m "feat: add Material Agent with retrieval and generation modes"
```

---

### Task 8: 视频渲染服务 & Celery Worker

**Files:**
- Create: `app/services/render_service.py`
- Create: `app/workers/__init__.py`
- Create: `app/workers/render_worker.py`
- Create: `app/core/celery_app.py`

**Interfaces:**
- Consumes: MoviePy, FFmpeg
- Produces: `RenderService.build_timeline(storyboard, material_selections) -> CompositeVideoClip`, `RenderService.render(clip, output_path) -> str`

- [ ] **Step 1: 编写 app/core/celery_app.py**

```python
from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "video_editing_agent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
)
```

- [ ] **Step 2: 编写 app/services/render_service.py**

```python
import os
import subprocess
import logging
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RenderService:
    """视频渲染服务 — 时间线构建 + 合成输出"""

    def __init__(self):
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    def build_timeline(
        self,
        storyboard: dict,
        material_selections: list[dict],
        bgm_path: str | None = None,
    ) -> CompositeVideoClip:
        """根据分镜和素材构建时间线"""
        clips = []
        target_w, target_h = map(int, settings.DEFAULT_RESOLUTION.split("x"))

        for shot in storyboard["shots"]:
            shot_index = shot["index"]
            material = self._find_material(shot_index, material_selections)
            if not material:
                logger.warning(f"No material found for shot {shot_index}, skipping")
                continue

            clip = self._load_clip(material, shot)
            if clip is None:
                continue

            # 裁剪到指定时长
            duration = shot["duration_sec"]
            clip = clip.with_duration(duration)

            # 缩放到目标分辨率
            clip = clip.resized(new_size=(target_w, target_h))

            # 转场处理（当前版本：直接拼接，转场效果后续迭代）
            clips.append(clip)

        if not clips:
            raise ValueError("No valid clips to compose")

        return concatenate_videoclips(clips, method="compose")

    def _find_material(self, shot_index: int, selections: list[dict]) -> dict | None:
        for s in selections:
            if s.get("shot_index") == shot_index:
                return s
        return None

    def _load_clip(self, material: dict, shot: dict):
        """加载并裁剪素材片段"""
        file_path = material.get("file_path") or material.get("file_name")
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None

        clip_range = material.get("clip_range", [0, shot["duration_sec"]])
        start, end = clip_range[0], clip_range[1]

        if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            clip = ImageClip(file_path)
            clip = clip.with_duration(end - start)
        else:
            clip = VideoFileClip(file_path)
            if start > 0 or end < clip.duration:
                clip = clip.subclipped(start, min(end, clip.duration))

        return clip

    def render(self, clip: CompositeVideoClip, output_filename: str) -> str:
        """渲染输出 MP4 文件"""
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

        clip.write_videofile(
            output_path,
            fps=settings.DEFAULT_FPS,
            codec="libx264",
            audio_codec="aac",
            temp_audio_file_path=settings.OUTPUT_DIR,
            threads=4,
            preset="medium",
        )

        self._validate_output(output_path)
        return output_path

    @staticmethod
    def _validate_output(file_path: str) -> bool:
        """校验输出文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Render output not found: {file_path}")
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Render complete: {file_path} ({size_mb:.1f} MB)")
        return True

    def add_bgm(self, video_path: str, bgm_path: str, volume: float = 0.3) -> str:
        """为视频添加背景音乐（使用 FFmpeg）"""
        output_path = video_path.replace(".mp4", "_bgm.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", bgm_path,
            "-filter_complex", f"[1:a]volume={volume}[a1];[0:a][a1]amix=inputs=2:duration=first",
            "-c:v", "copy",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
```

- [ ] **Step 3: 编写 app/workers/render_worker.py**

```python
import logging
from celery import Task
from app.core.celery_app import celery_app
from app.services.render_service import RenderService
from app.models.task import TaskStatus

logger = logging.getLogger(__name__)


class RenderTask(Task):
    """自定义 Celery Task，支持进度回调和状态更新"""
    _render_service = None

    @property
    def render_service(self):
        if self._render_service is None:
            self._render_service = RenderService()
        return self._render_service


@celery_app.task(bind=True, base=RenderTask, name="render_video")
def render_video(
    self,
    project_id: str,
    storyboard: dict,
    material_selections: list[dict],
    bgm_path: str | None = None,
) -> dict:
    """异步渲染视频任务"""
    try:
        self.update_state(state="PROCESSING", meta={"progress_pct": 10})

        # 构建时间线
        service = self.render_service
        clip = service.build_timeline(storyboard, material_selections, bgm_path)

        self.update_state(state="RENDERING", meta={"progress_pct": 50})

        # 渲染
        output_filename = f"{project_id}_{self.request.id}.mp4"
        output_path = service.render(clip, output_filename)

        # 添加 BGM
        if bgm_path and os.path.exists(bgm_path):
            self.update_state(state="RENDERING", meta={"progress_pct": 80})
            output_path = service.add_bgm(output_path, bgm_path)

        # 清理
        clip.close()

        return {
            "status": "completed",
            "output_path": output_path,
            "progress_pct": 100,
        }

    except Exception as e:
        logger.exception(f"Render failed for project {project_id}")
        return {
            "status": "failed",
            "error": str(e),
            "progress_pct": 0,
        }
```

- [ ] **Step 4: Commit**

```bash
git add app/services/render_service.py app/workers/ app/core/celery_app.py
git commit -m "feat: add video render service and Celery async worker"
```

---

### Task 9: 剪辑 Agent — 时间线编排 & 任务调度

**Files:**
- Create: `app/agents/editor.py`
- Create: `tests/test_editor.py`

**Interfaces:**
- Consumes: `RenderService` (from Task 8), Celery app
- Produces: `EditorAgent.submit_edit(project_id, storyboard, material_selections) -> task_id`

- [ ] **Step 1: 编写测试 tests/test_editor.py**

```python
import pytest
from unittest.mock import patch, MagicMock
from app.agents.editor import EditorAgent


class TestEditorAgent:
    @pytest.fixture
    def storyboard(self):
        return {
            "shots": [
                {"index": 1, "description": "test shot", "duration_sec": 3, "transition": "硬切"},
            ]
        }

    @pytest.fixture
    def material_selections(self):
        return [{"shot_index": 1, "file_path": "/fake/test.mp4", "clip_range": [0, 3]}]

    def test_submit_edit_returns_task_id(self, storyboard, material_selections):
        with patch("app.agents.editor.render_video") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id="celery-task-123"))

            agent = EditorAgent()
            result = agent.submit_edit(
                project_id="proj_001",
                storyboard=storyboard,
                material_selections=material_selections,
            )
            assert result == "celery-task-123"
            mock_task.delay.assert_called_once()

    def test_get_status_returns_task_info(self):
        with patch("app.agents.editor.AsyncResult") as MockResult:
            mock_result = MagicMock()
            mock_result.state = "RENDERING"
            mock_result.info = {"progress_pct": 65}
            MockResult.return_value = mock_result

            agent = EditorAgent()
            status = agent.get_task_status("task_001")
            assert status["status"] == "RENDERING"
            assert status["progress_pct"] == 65
```

- [ ] **Step 2: 运行测试验证失败**

```bash
pytest tests/test_editor.py -v
# Expected: FAIL
```

- [ ] **Step 3: 编写 app/agents/editor.py**

```python
import logging
import os
from celery.result import AsyncResult
from app.workers.render_worker import render_video

logger = logging.getLogger(__name__)


class EditorAgent:
    """剪辑 Agent — 提交渲染任务、查询状态"""

    def submit_edit(
        self,
        project_id: str,
        storyboard: dict,
        material_selections: list[dict],
        bgm_path: str | None = None,
    ) -> str:
        """提交异步剪辑任务，返回 task_id"""
        task = render_video.delay(
            project_id=project_id,
            storyboard=storyboard,
            material_selections=material_selections,
            bgm_path=bgm_path,
        )
        logger.info(f"Edit task submitted: {task.id} for project {project_id}")
        return task.id

    def get_task_status(self, task_id: str) -> dict:
        """查询渲染任务状态"""
        result = AsyncResult(task_id, app=render_video.app)

        response = {
            "task_id": task_id,
            "status": result.state.lower(),
            "progress_pct": 0,
            "output_path": None,
            "error": None,
        }

        if result.ready():
            if result.successful():
                data = result.result
                response["status"] = "completed"
                response["progress_pct"] = 100
                response["output_path"] = data.get("output_path") if isinstance(data, dict) else None
            else:
                response["status"] = "failed"
                response["error"] = str(result.info) if result.info else "Unknown error"
        elif result.state == "PROGRESS":
            meta = result.info or {}
            response["progress_pct"] = meta.get("progress_pct", 0)
        elif result.state == "STARTED":
            response["progress_pct"] = 5

        return response

    def get_output_url(self, task_id: str) -> dict:
        """获取渲染完成的视频访问信息"""
        status = self.get_task_status(task_id)
        if status["status"] != "completed":
            return {"ready": False, "status": status["status"]}

        output_path = status.get("output_path")
        if output_path and os.path.exists(output_path):
            return {
                "ready": True,
                "file_path": output_path,
                "file_size_bytes": os.path.getsize(output_path),
            }
        return {"ready": False, "error": "Output file not found"}
```

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_editor.py -v
# Expected: ALL PASS
```

- [ ] **Step 5: Commit**

```bash
git add app/agents/editor.py tests/test_editor.py
git commit -m "feat: add Editor Agent with async task submission and status query"
```

---

### Task 10: API 端点 — Projects & Story

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/projects.py`
- Create: `app/api/story.py`
- Create: `app/api/deps.py`

**Interfaces:**
- Produces: FastAPI routers: `projects.router`, `story.router`

- [ ] **Step 1: 编写 app/api/deps.py**

```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.llm_client import LLMClient
from app.agents.director import DirectorAgent
from app.agents.material import MaterialAgent
from app.agents.editor import EditorAgent


async def get_llm_client():
    return LLMClient()


def get_director_agent(llm: LLMClient = Depends(get_llm_client)):
    return DirectorAgent(llm_client=llm)


def get_material_agent(llm: LLMClient = Depends(get_llm_client)):
    return MaterialAgent(llm_client=llm)


def get_editor_agent():
    return EditorAgent()
```

- [ ] **Step 2: 编写 app/api/projects.py**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project, IndexStatus
import uuid
from datetime import datetime

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
```

- [ ] **Step 3: 编写 app/api/story.py**

```python
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
```

- [ ] **Step 4: 在 app/main.py 中注册路由**

```python
# 在 create_app() 中添加:
from app.api.projects import router as projects_router
from app.api.story import router as story_router

app.include_router(projects_router)
app.include_router(story_router)
```

- [ ] **Step 5: Commit**

```bash
git add app/api/ app/main.py
git commit -m "feat: add API endpoints for projects and story generation"
```

---

### Task 11: API 端点 — Materials & Edit

**Files:**
- Create: `app/api/materials.py`
- Create: `app/api/edit.py`

**Interfaces:**
- Produces: FastAPI routers: `materials.router`, `edit.router`

- [ ] **Step 1: 编写 app/api/materials.py**

```python
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
```

- [ ] **Step 2: 编写 app/api/edit.py**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from app.core.database import get_db
from app.models.project import Project
from app.models.task import EditTask, TaskStatus
from app.agents.editor import EditorAgent
from app.api.deps import get_editor_agent
import uuid

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
```

- [ ] **Step 3: 在 app/main.py 中注册新路由**

```python
from app.api.materials import router as materials_router
from app.api.edit import router as edit_router

app.include_router(materials_router)
app.include_router(edit_router)
```

- [ ] **Step 4: Commit**

```bash
git add app/api/materials.py app/api/edit.py app/main.py
git commit -m "feat: add API endpoints for material matching and video editing"
```

---

### Task 12: 集成测试 & 完善

**Files:**
- Create: `tests/test_api_integration.py`
- Create: `README.md`
- Create: `.env.example`

- [ ] **Step 1: 编写 tests/test_api_integration.py**

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient
from app.main import create_app


class TestAPI:
    @pytest.fixture
    async def async_client(self):
        app = create_app()
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, async_client):
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_create_project_requires_existing_dir(self, async_client):
        response = await async_client.post(
            "/api/v1/projects",
            json={
                "name": "test",
                "material_source_dir": "/nonexistent/path",
            },
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_get_nonexistent_project_returns_404(self, async_client):
        response = await async_client.get("/api/v1/projects/nonexistent-id")
        assert response.status_code == 404
```

- [ ] **Step 2: 编写 .env.example**

```
# 火山方舟
VOLCANO_ARK_API_KEY=your_api_key_here
VOLCANO_ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
VOLCANO_ARK_MODEL=doubao-vision-pro-32k

# 数据库
DATABASE_URL=postgresql+asyncpg://vagent:vagent_secret@localhost:5432/vagent
DATABASE_URL_SYNC=postgresql+psycopg2://vagent:vagent_secret@localhost:5432/vagent

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# 输出
OUTPUT_DIR=./output

# 中文字体路径
FONT_PATH=C:/Windows/Fonts/simhei.ttf
```

- [ ] **Step 3: 编写 README.md 核心内容**

# 基于多模态大模型的智能视频剪辑 Agent

## 快速启动

```bash
# 1. 启动依赖服务
docker-compose up -d

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入火山方舟 API Key

# 3. 安装依赖
pip install -r requirements.txt

# 4. 数据库迁移
alembic upgrade head

# 5. 启动 API
uvicorn app.main:app --reload --port 8000

# 6. 启动 Celery Worker（新终端）
celery -A app.core.celery_app worker --loglevel=info --concurrency=1
```

## API 使用流程

```bash
# 1. 创建项目
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "旅行vlog", "material_source_dir": "/path/to/materials"}'

# 2. 生成分镜
curl -X POST http://localhost:8000/api/v1/projects/{id}/story \
  -H "Content-Type: application/json" \
  -d '{"prompt": "旅行素材30秒快剪", "style": "快节奏", "total_duration_sec": 30}'

# 3. 匹配素材
curl -X POST http://localhost:8000/api/v1/projects/{id}/materials \
  -H "Content-Type: application/json" \
  -d '{"mode": "retrieval", "storyboard": {...}}'

# 4. 提交剪辑
curl -X POST http://localhost:8000/api/v1/projects/{id}/edit \
  -H "Content-Type: application/json" \
  -d '{"storyboard": {...}, "material_selections": [...]}'

# 5. 查询进度
curl "http://localhost:8000/api/v1/projects/{id}/status?task_id={task_id}"

# 6. 获取结果
curl "http://localhost:8000/api/v1/projects/{id}/result?task_id={task_id}"
```

- [ ] **Step 4: 运行集成测试**

```bash
pytest tests/test_api_integration.py -v
# Expected: ALL PASS
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_integration.py README.md .env.example
git commit -m "feat: add integration tests, README, and env template"
```

---

### Task 13: 素材索引异步任务（补充）

**Files:**
- Modify: `app/services/index_service.py`
- Create: `app/workers/index_worker.py`

**Interfaces:**
- Produces: `index_project.delay(project_id, dir_path)` → Celery task that scans, extracts keyframes, and indexes all media files

- [ ] **Step 1: 编写 app/workers/index_worker.py**

```python
import os
import tempfile
import logging
from celery import Task
from app.core.celery_app import celery_app
from app.services.file_service import FileService
from app.services.index_service import IndexService
from app.models.project import IndexStatus

logger = logging.getLogger(__name__)


class IndexTask(Task):
    _index_service = None

    @property
    def index_service(self):
        if self._index_service is None:
            self._index_service = IndexService()
        return self._index_service


@celery_app.task(bind=True, base=IndexTask, name="index_project")
def index_project(self, project_id: str, dir_path: str) -> dict:
    """异步索引项目素材目录"""
    try:
        files = FileService.scan_directory(dir_path)
        indexed = 0
        errors = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            for file_info in files:
                self.update_state(
                    state="PROGRESS",
                    meta={"indexed": indexed, "total": len(files)},
                )

                try:
                    keyframe_path = None
                    if file_info["file_type"] == "video":
                        keyframe_path = os.path.join(
                            tmpdir, f"{file_info['file_name']}_kf.jpg"
                        )
                        if FileService.extract_keyframe(file_info["file_path"], keyframe_path, at_second=1.0):
                            self.index_service.index_single(project_id, file_info, keyframe_path)
                            indexed += 1

                    elif file_info["file_type"] == "image":
                        self.index_service.index_single(project_id, file_info, file_info["file_path"])
                        indexed += 1

                except Exception as e:
                    logger.error(f"Failed to index {file_info['file_path']}: {e}")
                    errors += 1
                    continue

        return {
            "status": "completed",
            "indexed": indexed,
            "total": len(files),
            "errors": errors,
        }

    except Exception as e:
        logger.exception(f"Indexing failed for project {project_id}")
        return {"status": "failed", "error": str(e)}
```

- [ ] **Step 2: Commit**

```bash
git add app/workers/index_worker.py
git commit -m "feat: add async material indexing Celery worker"
```

---

## 实现顺序建议

按 Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 顺序执行。Phase 依赖关系：

```
Foundation (1-2) → Core Services (3-4) → Agents (5-7-9) + Render (8) → API (10-11) → Integration (12) → Index Worker (13)
                                     ↑
                              Index Service (6)
```

每个 Task 完成后运行测试、commit，再进入下一个 Task。
