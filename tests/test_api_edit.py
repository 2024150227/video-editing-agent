"""Tests for app/api/edit.py — Edit API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import create_app
from app.api.deps import get_editor_agent, get_db


@pytest.fixture
def app_with_overrides():
    app = create_app()
    return app


@pytest.fixture
def client(app_with_overrides):
    return TestClient(app_with_overrides)


@pytest.fixture
def mock_editor_agent():
    """Return a mock EditorAgent with canned responses."""
    agent = MagicMock()
    agent.submit_edit.return_value = "task-abc-123"

    agent.get_task_status.return_value = {
        "status": "completed",
        "progress_pct": 100,
        "output_path": "/output/final.mp4",
        "error": None,
    }

    agent.get_output_url.return_value = {
        "ready": True,
        "file_path": "/output/final.mp4",
        "file_size_bytes": 1048576,
        "error": None,
    }

    return agent


class TestStartEdit:
    """POST /api/v1/projects/{project_id}/edit"""

    VALID_BODY = {
        "storyboard": {"shots": [{"index": 0, "description": "test"}]},
        "material_selections": [
            {"shot_index": 0, "material_id": "mat-001", "clip_range": [0.0, 5.0]}
        ],
        "bgm_path": "/music/bg.mp3",
        "subtitles": "Hello world",
    }

    @pytest.fixture(autouse=True)
    def setup_overrides(self, app_with_overrides, mock_editor_agent):
        overrides = {}
        overrides[get_editor_agent] = lambda: mock_editor_agent
        app_with_overrides.dependency_overrides.update(overrides)
        yield
        for dep in overrides:
            app_with_overrides.dependency_overrides.pop(dep, None)

    def _mock_db(self, app_with_overrides, project_exists=True, project_id="proj-001", existing_task=False):
        """Set up a mock DB via dependency_overrides.

        Uses MagicMock for the execute result since scalar_one_or_none()
        is a synchronous method on SQLAlchemy's Result class.
        """
        from app.api.edit import get_db as edit_get_db

        mock_session = AsyncMock()
        project_result = MagicMock()
        task_result = MagicMock()

        if project_exists:
            mock_project = MagicMock()
            mock_project.id = project_id
            project_result.scalar_one_or_none.return_value = mock_project
        else:
            project_result.scalar_one_or_none.return_value = None

        if existing_task:
            task_result.scalar_one_or_none.return_value = MagicMock()
        else:
            task_result.scalar_one_or_none.return_value = None

        async def mock_execute(stmt, **kwargs):
            stmt_str = str(stmt)
            if "FROM projects" in stmt_str or "FROM projects" in str(stmt.compile(compile_kwargs={"literal_binds": True})):
                pass
            if "projects" in stmt_str:
                return project_result
            else:
                return task_result

        mock_session.execute = mock_execute

        async def _override():
            yield mock_session

        app_with_overrides.dependency_overrides[edit_get_db] = _override
        return mock_session

    def test_start_edit_success(self, client, app_with_overrides):
        """200: successful edit submission."""
        self._mock_db(app_with_overrides, project_exists=True, project_id="proj-001", existing_task=False)

        resp = client.post("/api/v1/projects/proj-001/edit", json=self.VALID_BODY)
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert data["task_id"] == "task-abc-123"
        assert data["status"] == "queued"

    def test_start_edit_project_not_found(self, client, app_with_overrides):
        """404: project does not exist."""
        self._mock_db(app_with_overrides, project_exists=False, project_id="nonexistent")

        resp = client.post("/api/v1/projects/nonexistent/edit", json=self.VALID_BODY)
        assert resp.status_code == 404

    def test_start_edit_concurrent_task_blocked(self, client, app_with_overrides):
        """429: existing queued/processing task for same project."""
        self._mock_db(app_with_overrides, project_exists=True, project_id="proj-001", existing_task=True)

        resp = client.post("/api/v1/projects/proj-001/edit", json=self.VALID_BODY)
        assert resp.status_code == 429

    def test_start_edit_invalid_body(self, client, app_with_overrides):
        """422: missing required fields."""
        self._mock_db(app_with_overrides, project_exists=True, project_id="proj-001")

        resp = client.post("/api/v1/projects/proj-001/edit", json={})
        assert resp.status_code == 422

    def test_start_edit_without_bgm(self, client, app_with_overrides):
        """200: optional bgm_path can be omitted."""
        self._mock_db(app_with_overrides, project_exists=True, project_id="proj-001", existing_task=False)

        body = {
            "storyboard": {"shots": []},
            "material_selections": [],
        }
        resp = client.post("/api/v1/projects/proj-001/edit", json=body)
        assert resp.status_code == 200


class TestGetEditStatus:
    """GET /api/v1/projects/{project_id}/status?task_id=..."""

    @pytest.fixture(autouse=True)
    def setup_overrides(self, app_with_overrides, mock_editor_agent):
        overrides = {}
        overrides[get_editor_agent] = lambda: mock_editor_agent
        app_with_overrides.dependency_overrides.update(overrides)
        yield
        for dep in overrides:
            app_with_overrides.dependency_overrides.pop(dep, None)

    def test_get_status_completed(self, client, app_with_overrides):
        """200: get completed task status."""
        resp = client.get(
            "/api/v1/projects/proj-001/status",
            params={"task_id": "task-abc-123"},
        )
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert data["task_id"] == "task-abc-123"
        assert data["status"] == "completed"
        assert data["progress_pct"] == 100

    def test_get_status_in_progress(self, client, app_with_overrides, mock_editor_agent):
        """200: get in-progress task status."""
        mock_editor_agent.get_task_status.return_value = {
            "status": "PROGRESS",
            "progress_pct": 45,
            "output_path": None,
            "error": None,
        }

        resp = client.get(
            "/api/v1/projects/proj-001/status",
            params={"task_id": "task-in-progress"},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "PROGRESS"
        assert data["progress_pct"] == 45


class TestGetEditResult:
    """GET /api/v1/projects/{project_id}/result?task_id=..."""

    @pytest.fixture(autouse=True)
    def setup_overrides(self, app_with_overrides, mock_editor_agent):
        overrides = {}
        overrides[get_editor_agent] = lambda: mock_editor_agent
        app_with_overrides.dependency_overrides.update(overrides)
        yield
        for dep in overrides:
            app_with_overrides.dependency_overrides.pop(dep, None)

    def test_get_result_ready(self, client, app_with_overrides):
        """200: result ready with file info."""
        resp = client.get(
            "/api/v1/projects/proj-001/result",
            params={"task_id": "task-abc-123"},
        )
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert data["ready"] is True
        assert data["file_path"] == "/output/final.mp4"
        assert data["file_size_bytes"] == 1048576

    def test_get_result_not_ready(self, client, app_with_overrides, mock_editor_agent):
        """200: result not ready yet."""
        mock_editor_agent.get_output_url.return_value = {
            "ready": False,
            "file_path": None,
            "file_size_bytes": None,
            "error": "Output file not found",
        }

        resp = client.get(
            "/api/v1/projects/proj-001/result",
            params={"task_id": "task-pending"},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["ready"] is False
