"""Tests for app/api/materials.py — Materials API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import create_app
from app.api.deps import get_material_agent, get_db


@pytest.fixture
def app_with_overrides():
    """Create app and return both app and a helper to register overrides."""
    app = create_app()
    return app


@pytest.fixture
def client(app_with_overrides):
    return TestClient(app_with_overrides)


@pytest.fixture
def mock_material_agent():
    """Return a mock MaterialAgent that returns a canned match result."""
    agent = AsyncMock()
    agent.match_materials.return_value = [
        {
            "shot_index": 0,
            "candidates": [
                {
                    "material_id": "mat-001",
                    "file_name": "video_001.mp4",
                    "clip_range": [0.0, 5.0],
                    "score": 0.92,
                    "reason": "语义匹配得分 0.92",
                },
                {
                    "material_id": "mat-002",
                    "file_name": "video_002.mp4",
                    "clip_range": [0.0, 5.0],
                    "score": 0.85,
                    "reason": "语义匹配得分 0.85",
                },
            ],
            "suggestion": None,
        },
        {
            "shot_index": 1,
            "candidates": [
                {
                    "material_id": "mat-003",
                    "file_name": "generated_shot_1.png",
                    "clip_range": [0.0, 3.0],
                    "score": 1.0,
                    "reason": "AI 生成素材",
                },
            ],
            "suggestion": None,
        },
    ]
    return agent


class TestMatchMaterials:
    """POST /api/v1/projects/{project_id}/materials"""

    @pytest.fixture(autouse=True)
    def setup_overrides(self, app_with_overrides, mock_material_agent):
        """Set up dependency overrides before each test and clean up after."""
        overrides = {}
        overrides[get_material_agent] = lambda: mock_material_agent
        app_with_overrides.dependency_overrides.update(overrides)
        yield
        for dep in overrides:
            app_with_overrides.dependency_overrides.pop(dep, None)

    def _make_mock_db(self, app_with_overrides, project_exists=True, project_id="proj-001"):
        """Create a mock DB session and register it as an override.

        Uses MagicMock for the execute result since scalar_one_or_none()
        is a synchronous method on SQLAlchemy's Result class.
        """
        from app.api.materials import get_db as materials_get_db

        mock_session = AsyncMock()
        mock_exec_result = MagicMock()

        if project_exists:
            mock_project = MagicMock()
            mock_project.id = project_id
            mock_exec_result.scalar_one_or_none.return_value = mock_project
        else:
            mock_exec_result.scalar_one_or_none.return_value = None

        async def mock_execute(*args, **kwargs):
            return mock_exec_result

        mock_session.execute = mock_execute

        async def _override():
            yield mock_session

        app_with_overrides.dependency_overrides[materials_get_db] = _override
        return mock_session

    def test_match_materials_retrieval_success(self, client, app_with_overrides):
        """200: successful material matching in retrieval mode."""
        self._make_mock_db(app_with_overrides, project_exists=True, project_id="proj-001")

        body = {
            "mode": "retrieval",
            "storyboard": {
                "shots": [
                    {"index": 0, "description": "夕阳下的海滩", "duration_sec": 5.0},
                    {"index": 1, "description": "情侣牵手漫步", "duration_sec": 3.0},
                ]
            },
        }

        resp = client.post("/api/v1/projects/proj-001/materials", json=body)
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

        first_shot = data[0]
        assert first_shot["shot_index"] == 0
        assert len(first_shot["candidates"]) == 2
        assert first_shot["candidates"][0]["material_id"] == "mat-001"
        assert first_shot["candidates"][0]["score"] == 0.92

    def test_match_materials_generation_success(self, client, app_with_overrides):
        """200: successful material matching in generation mode."""
        self._make_mock_db(app_with_overrides, project_exists=True, project_id="proj-001")

        body = {
            "mode": "generation",
            "storyboard": {
                "shots": [
                    {"index": 0, "description": "夕阳下的海滩", "duration_sec": 5.0},
                ]
            },
        }

        resp = client.post("/api/v1/projects/proj-001/materials", json=body)
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert len(data) == 2  # mock returns 2 shots regardless
        assert data[1]["candidates"][0]["reason"] == "AI 生成素材"

    def test_match_materials_project_not_found(self, client, app_with_overrides):
        """404: project does not exist."""
        self._make_mock_db(app_with_overrides, project_exists=False, project_id="nonexistent")

        body = {
            "mode": "retrieval",
            "storyboard": {"shots": []},
        }

        resp = client.post("/api/v1/projects/nonexistent/materials", json=body)
        assert resp.status_code == 404
        assert "项目不存在" in resp.text

    def test_match_materials_invalid_mode(self, client, app_with_overrides):
        """400: invalid mode."""
        self._make_mock_db(app_with_overrides, project_exists=True, project_id="proj-001")

        body = {
            "mode": "invalid_mode",
            "storyboard": {"shots": []},
        }

        resp = client.post("/api/v1/projects/proj-001/materials", json=body)
        assert resp.status_code == 400
        assert "mode" in resp.text

    def test_match_materials_with_overrides(self, client, app_with_overrides):
        """200: with material_overrides."""
        mock_agent = app_with_overrides.dependency_overrides[get_material_agent]()
        self._make_mock_db(app_with_overrides, project_exists=True, project_id="proj-001")

        body = {
            "mode": "retrieval",
            "storyboard": {"shots": [{"index": 0, "description": "test", "duration_sec": 3.0}]},
            "material_overrides": {"0": "custom_video.mp4"},
        }

        resp = client.post("/api/v1/projects/proj-001/materials", json=body)
        assert resp.status_code == 200

        # Verify the mock was called with the right arguments
        mock_agent.match_materials.assert_called_once()
        call_kwargs = mock_agent.match_materials.call_args[1]
        assert call_kwargs["material_overrides"] == {"0": "custom_video.mp4"}
