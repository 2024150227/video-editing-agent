"""Integration tests for the Video Editing Agent API (LangGraph-based).

Uses mocked graph + DB session to verify routing, validation, error handling,
and response schemas without real infrastructure.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import create_app
from app.api.deps import get_graph


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


def _register_db_override(app, mock_session):
    """Register mock DB session on all API modules that import get_db."""
    from app.api import projects as pm
    from app.api import story as sm
    from app.api import materials as mm
    from app.api import edit as em

    async def _override():
        yield mock_session

    for mod in [pm, sm, mm, em]:
        fn = getattr(mod, "get_db", None)
        if fn:
            app.dependency_overrides[fn] = _override


def _mock_graph_for_story(storyboard):
    """Graph that returns state after director_node's interrupt."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "storyboard": storyboard,
        "current_step": "storyboard_review",
    })
    return graph


def _mock_graph_for_materials(matches):
    """Graph that returns state after material_node's interrupt."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "material_matches": matches,
        "current_step": "materials_approved",
    })
    return graph


def _mock_graph_for_edit(task_id="task-001"):
    """Graph that returns state after editor_node completes."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "task_id": task_id,
        "task_status": {"status": "completed", "progress_pct": 100},
        "current_step": "done",
    })
    return graph


def _make_db(project_exists=True):
    """Create a mock AsyncSession with configurable results."""
    from app.models.project import IndexStatus

    project = MagicMock()
    project.id = "proj-001"
    project.name = "test"
    project.material_source_dir = "/tmp/m"
    project.index_status = IndexStatus.READY
    project.created_at = MagicMock()
    project.created_at.isoformat.return_value = "2025-01-01T00:00:00"

    session = AsyncMock()
    exec_result = MagicMock()

    def _set_results(*values):
        exec_result.scalar_one_or_none = MagicMock(side_effect=list(values))

    _set_results(project if project_exists else None)
    session._set_results = _set_results

    async def _execute(*a, **kw):
        return exec_result

    session.execute = _execute

    async def _refresh(obj, *a, **kw):
        if getattr(obj, "id", None) is None:
            obj.id = "auto-id"
        if getattr(obj, "created_at", None) is None:
            from datetime import datetime
            obj.created_at = datetime(2025, 1, 1)

    session.refresh = AsyncMock(side_effect=_refresh)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    return session


# ── Health ────────────────────────────────────────────────

class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Projects ──────────────────────────────────────────────

class TestProjects:
    def test_create_project_bad_dir(self, client, app):
        db = _make_db(project_exists=True)
        _register_db_override(app, db)
        resp = client.post("/api/v1/projects", json={
            "name": "test", "material_source_dir": "/nonexistent/path",
        })
        assert resp.status_code == 400

    def test_create_project_success(self, client, app, tmp_path):
        import os
        d = str(tmp_path / "materials")
        os.makedirs(d)
        db = _make_db(project_exists=True)
        _register_db_override(app, db)
        resp = client.post("/api/v1/projects", json={
            "name": "test", "material_source_dir": d,
        })
        assert resp.status_code == 201
        assert resp.json()["name"] == "test"

    def test_get_project_404(self, client, app):
        db = _make_db(project_exists=False)
        _register_db_override(app, db)
        resp = client.get("/api/v1/projects/nope")
        assert resp.status_code == 404

    def test_delete_project_204(self, client, app):
        db = _make_db(project_exists=True)
        _register_db_override(app, db)
        resp = client.delete("/api/v1/projects/proj-001")
        assert resp.status_code == 204


# ── Story ─────────────────────────────────────────────────

class TestStory:
    def test_storyboard_success(self, client, app):
        db = _make_db(project_exists=True)
        _register_db_override(app, db)

        graph = _mock_graph_for_story({
            "style": "快节奏", "total_duration_sec": 30, "bgm_mood": "动感",
            "shots": [
                {"index": 1, "duration_sec": 3, "description": "test", "shot_type": "wide", "transition": "硬切", "mood_words": []}
            ],
        })
        app.dependency_overrides[get_graph] = lambda: graph

        resp = client.post("/api/v1/projects/proj-001/story", json={
            "prompt": "test", "style": "快节奏", "total_duration_sec": 30,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["style"] == "快节奏"
        assert len(data["shots"]) > 0

    def test_storyboard_404(self, client, app):
        db = _make_db(project_exists=False)
        _register_db_override(app, db)
        graph = _mock_graph_for_story({})
        app.dependency_overrides[get_graph] = lambda: graph

        resp = client.post("/api/v1/projects/nope/story", json={
            "prompt": "test", "style": "", "total_duration_sec": 15,
        })
        assert resp.status_code == 404


# ── Materials ─────────────────────────────────────────────

class TestMaterials:
    def test_materials_success(self, client, app):
        db = _make_db(project_exists=True)
        _register_db_override(app, db)

        graph = _mock_graph_for_materials([
            {"shot_index": 1, "candidates": [
                {"material_id": "m1", "file_name": "v.mp4", "clip_range": [0,5], "score": 0.9, "reason": "match"}
            ]},
        ])
        app.dependency_overrides[get_graph] = lambda: graph

        resp = client.post("/api/v1/projects/proj-001/materials", json={
            "mode": "retrieval",
            "storyboard": {"shots": [{"index": 1, "description": "test", "duration_sec": 5}]},
        })
        assert resp.status_code == 200
        assert resp.json()[0]["candidates"][0]["material_id"] == "m1"

    def test_materials_404(self, client, app):
        db = _make_db(project_exists=False)
        _register_db_override(app, db)
        graph = _mock_graph_for_materials([])
        app.dependency_overrides[get_graph] = lambda: graph
        resp = client.post("/api/v1/projects/nope/materials", json={
            "mode": "retrieval", "storyboard": {"shots": []},
        })
        assert resp.status_code == 404

    def test_materials_bad_mode(self, client, app):
        db = _make_db(project_exists=True)
        _register_db_override(app, db)
        graph = _mock_graph_for_materials([])
        app.dependency_overrides[get_graph] = lambda: graph
        resp = client.post("/api/v1/projects/proj-001/materials", json={
            "mode": "bad", "storyboard": {"shots": []},
        })
        assert resp.status_code == 400


# ── Edit ──────────────────────────────────────────────────

class TestEdit:
    def test_edit_success(self, client, app):
        db = _make_db(project_exists=True)
        db._set_results(MagicMock(id="proj-001"), None)  # project exists, no task
        _register_db_override(app, db)

        graph = _mock_graph_for_edit("task-abc")
        app.dependency_overrides[get_graph] = lambda: graph

        resp = client.post("/api/v1/projects/proj-001/edit", json={
            "storyboard": {"shots": [{"index": 1, "duration_sec": 3}]},
            "material_selections": [{"shot_index": 1, "file_path": "/t.mp4"}],
        })
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "task-abc"

    def test_edit_404(self, client, app):
        db = _make_db(project_exists=False)
        _register_db_override(app, db)
        graph = _mock_graph_for_edit()
        app.dependency_overrides[get_graph] = lambda: graph
        resp = client.post("/api/v1/projects/nope/edit", json={
            "storyboard": {"shots": []}, "material_selections": [],
        })
        assert resp.status_code == 404

    def test_edit_429(self, client, app):
        db = _make_db(project_exists=True)
        db._set_results(MagicMock(id="proj-001"), MagicMock(id="existing"))
        _register_db_override(app, db)
        graph = _mock_graph_for_edit()
        app.dependency_overrides[get_graph] = lambda: graph
        resp = client.post("/api/v1/projects/proj-001/edit", json={
            "storyboard": {"shots": []}, "material_selections": [],
        })
        assert resp.status_code == 429


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
