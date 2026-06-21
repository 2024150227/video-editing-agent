"""Tests for app/api/edit.py — Edit API with LangGraph."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import create_app
from app.api.deps import get_graph


def _mock_graph_for_edit(task_id="task-001", status="completed"):
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "task_id": task_id,
        "task_status": {"status": status, "progress_pct": 100},
        "current_step": "done",
    })
    return mock_graph


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestStartEdit:
    @pytest.fixture(autouse=True)
    def setup(self, app):
        from app.api import edit as mod

        # Graph override
        graph = _mock_graph_for_edit()
        app.dependency_overrides[get_graph] = lambda: graph

        # DB override — project exists, no existing task
        db_session = AsyncMock()
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none = MagicMock(side_effect=[
            MagicMock(id="proj-001"),  # project exists
            None,                       # no existing task
        ])
        db_session.execute = AsyncMock(return_value=mock_exec)
        db_session.add = MagicMock()
        db_session.commit = AsyncMock()

        async def _db_override():
            yield db_session

        app.dependency_overrides[mod.get_db] = _db_override
        self._db = db_session
        self._mock_exec = mock_exec
        yield
        app.dependency_overrides.clear()

    def test_edit_success(self, client):
        resp = client.post("/api/v1/projects/proj-001/edit", json={
            "storyboard": {"shots": [{"index": 1, "duration_sec": 3}]},
            "material_selections": [{"shot_index": 1, "file_path": "/t.mp4"}],
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["task_id"] == "task-001"

    def test_edit_404(self, client):
        self._mock_exec.scalar_one_or_none = MagicMock(return_value=None)
        resp = client.post("/api/v1/projects/nope/edit", json={
            "storyboard": {"shots": []}, "material_selections": [],
        })
        assert resp.status_code == 404

    def test_edit_429(self, client):
        self._mock_exec.scalar_one_or_none = MagicMock(side_effect=[
            MagicMock(id="proj-001"), MagicMock(id="existing-task"),
        ])
        resp = client.post("/api/v1/projects/proj-001/edit", json={
            "storyboard": {"shots": []}, "material_selections": [],
        })
        assert resp.status_code == 429

    def test_edit_without_bgm(self, client):
        resp = client.post("/api/v1/projects/proj-001/edit", json={
            "storyboard": {"shots": [{"index": 1, "duration_sec": 3}]},
            "material_selections": [{"shot_index": 1, "file_path": "/t.mp4"}],
        })
        assert resp.status_code == 200


class TestEditStatus:
    def test_status(self):
        with patch("app.api.edit.EditorAgent") as MockEditor:
            mock = MagicMock()
            mock.get_task_status.return_value = {
                "status": "rendering", "progress_pct": 50,
            }
            MockEditor.return_value = mock
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/projects/p1/status?task_id=t1")
            assert resp.status_code == 200
            assert resp.json()["progress_pct"] == 50


class TestEditResult:
    def test_result_ready(self):
        with patch("app.api.edit.EditorAgent") as MockEditor:
            mock = MagicMock()
            mock.get_output_url.return_value = {
                "ready": True, "file_path": "/out/v.mp4", "file_size_bytes": 100,
            }
            MockEditor.return_value = mock
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/projects/p1/result?task_id=t1")
            assert resp.status_code == 200
            assert resp.json()["ready"] is True

    def test_result_not_ready(self):
        with patch("app.api.edit.EditorAgent") as MockEditor:
            mock = MagicMock()
            mock.get_output_url.return_value = {"ready": False}
            MockEditor.return_value = mock
            app = create_app()
            client = TestClient(app)
            resp = client.get("/api/v1/projects/p1/result?task_id=t1")
            assert resp.status_code == 200
            assert resp.json()["ready"] is False
