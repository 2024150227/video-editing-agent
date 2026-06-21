"""Integration tests for the Video Editing Agent API.

These tests use dependency overrides (mocked DB sessions, mocked agents)
to verify routing, request validation, error handling, and response schemas
for every endpoint without requiring real infrastructure (PostgreSQL, Redis, etc.).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_db_session():
    """Create a base AsyncMock session with a configurable execute().

    The returned session has:
    - ``set_project_exists(val)`` - control whether project lookups succeed.
    - ``add_spy`` / ``flush_spy`` / ``commit_spy`` / ``refresh_spy`` - track
      calls for assertion, and the default ``refresh`` side-effect sets ``.id``
      on the refreshed object so the response can include it.
    - ``execute`` returns results from a configurable list of per-call results.
      By default, just one result (the project or None). Append more with
      ``push_exec_result(scalar_return_value)``.
    """
    from app.models.project import IndexStatus

    project = MagicMock()
    project.id = "proj-001"
    project.name = "test-project"
    project.material_source_dir = "/tmp/materials"
    project.index_status = IndexStatus.READY

    class _FakeDateTime:
        def isoformat(self):
            return "2025-01-01T00:00:00"
    project.created_at = _FakeDateTime()

    # Build a callable that returns different results per execute call
    exec_results = []
    _default_result = MagicMock()
    _default_result.scalar_one_or_none.return_value = project
    exec_results.append(_default_result)
    _call_index = 0

    async def _next_exec_result(*args, **kwargs):
        nonlocal _call_index
        if _call_index < len(exec_results):
            result = exec_results[_call_index]
        else:
            # Last defined result repeats
            result = exec_results[-1] if exec_results else _default_result
        _call_index += 1
        return result

    session = AsyncMock()

    # Let each test push extra execute results for multi-query endpoints (e.g.
    # edit/: project lookup + task lookup). Index 0 is the project-check result.
    def push_exec_result(scalar_return_value):
        r = MagicMock()
        r.scalar_one_or_none.return_value = scalar_return_value
        exec_results.append(r)

    session.push_exec_result = push_exec_result
    session.execute = _next_exec_result

    # refresh side effect: set .id and .created_at on the refreshed object,
    # and populate shots relationship for Storyboard objects
    _added_objects = []

    def _add_sync(obj, *a, **kw):
        _added_objects.append(obj)

    session.add = MagicMock(side_effect=_add_sync)

    async def _refresh_side_effect(obj, *a, **kw):
        from datetime import datetime
        if getattr(obj, "id", None) is None:
            obj.id = "auto-" + str(hash(str(obj)) % 10**8)
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime(2025, 1, 1)
        # Handle storyboard.shots — collect Shot objects added to session
        if type(obj).__name__ == "Storyboard":
            from app.models.storyboard import Shot
            obj.shots = [s for s in _added_objects if isinstance(s, Shot)]

    session.refresh = AsyncMock(side_effect=_refresh_side_effect)

    # Toggle project existence (resets call index)
    def set_project_exists(val):
        nonlocal _call_index
        _call_index = 0
        _default_result.scalar_one_or_none.return_value = project if val else None
        # If there are extra results, set them back to the default too
        for r in exec_results:
            r.scalar_one_or_none.return_value = project if val else None

    session.set_project_exists = set_project_exists

    return session


@pytest.fixture
def mock_llm(app):
    """Prevent real LLM client instantiation by overriding get_llm_client."""
    mock = AsyncMock()
    mock.chat.return_value = "mocked response"
    from app.api.deps import get_llm_client
    app.dependency_overrides[get_llm_client] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_llm_client, None)


@pytest.fixture
def mock_director(app):
    """Mock DirectorAgent with a canned storyboard response."""
    agent = AsyncMock()
    agent.generate_storyboard.return_value = {
        "style": "快节奏",
        "total_duration_sec": 30,
        "bgm_mood": "欢快",
        "shots": [
            {
                "index": 0,
                "duration_sec": 5,
                "description": "海滩日出空镜",
                "shot_type": "远景",
                "camera_motion": "推",
                "transition": "硬切",
                "mood_words": ["温暖", "宁静"],
            },
            {
                "index": 1,
                "duration_sec": 4,
                "description": "人物奔跑特写",
                "shot_type": "特写",
                "camera_motion": "跟拍",
                "transition": "交叉溶解",
                "mood_words": ["动感", "活力"],
            },
        ],
    }
    from app.api.deps import get_director_agent
    app.dependency_overrides[get_director_agent] = lambda: agent
    yield agent
    app.dependency_overrides.pop(get_director_agent, None)


@pytest.fixture
def mock_material_agent(app):
    """Mock MaterialAgent with a canned match result."""
    agent = AsyncMock()
    agent.match_materials.return_value = [
        {
            "shot_index": 0,
            "candidates": [
                {
                    "material_id": "mat-001",
                    "file_name": "beach.mp4",
                    "clip_range": [0.0, 5.0],
                    "score": 0.92,
                    "reason": "语义匹配得分 0.92",
                }
            ],
            "suggestion": None,
        },
    ]
    from app.api.deps import get_material_agent
    app.dependency_overrides[get_material_agent] = lambda: agent
    yield agent
    app.dependency_overrides.pop(get_material_agent, None)


@pytest.fixture
def mock_editor_agent(app):
    """Mock EditorAgent with canned responses for edit/status/result."""
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
    from app.api.deps import get_editor_agent
    app.dependency_overrides[get_editor_agent] = lambda: agent
    yield agent
    app.dependency_overrides.pop(get_editor_agent, None)


def _override_db_on_all_modules(app, mock_session):
    """Override ``get_db`` on every API module that imports it.

    Returns a cleanup callable.
    """
    from app.api import projects as projects_mod
    from app.api import story as story_mod
    from app.api import materials as materials_mod
    from app.api import edit as edit_mod

    modules = [projects_mod, story_mod, materials_mod, edit_mod]
    saved = {}
    for mod in modules:
        fn = getattr(mod, "get_db", None)
        if fn is not None:
            saved[mod] = app.dependency_overrides.get(fn)

    async def _override():
        yield mock_session

    for mod in modules:
        fn = getattr(mod, "get_db", None)
        if fn is not None:
            app.dependency_overrides[fn] = _override

    def cleanup():
        for mod in modules:
            fn = getattr(mod, "get_db", None)
            if fn is not None:
                if mod in saved and saved[mod] is not None:
                    app.dependency_overrides[fn] = saved[mod]
                else:
                    app.dependency_overrides.pop(fn, None)

    return cleanup


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /health"""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class TestCreateProject:
    """POST /api/v1/projects"""

    def test_create_project_requires_existing_dir(self, client, app, mock_db_session):
        """400: material_source_dir must exist on disk."""
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            resp = client.post(
                "/api/v1/projects",
                json={
                    "name": "test",
                    "material_source_dir": "/nonexistent/path",
                },
            )
            assert resp.status_code == 400
            assert "不存在" in resp.text or "exist" in resp.text
        finally:
            cleanup()

    def test_create_project_success(self, client, app, mock_db_session, tmp_path):
        """201: valid project is created."""
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            existing_dir = str(tmp_path / "materials")
            import os
            os.makedirs(existing_dir)

            resp = client.post(
                "/api/v1/projects",
                json={
                    "name": "travel-vlog",
                    "material_source_dir": existing_dir,
                },
            )
            assert resp.status_code == 201, resp.text
            data = resp.json()
            assert data["name"] == "travel-vlog"
            assert "project_id" in data
            assert data["index_status"] == "pending"
        finally:
            cleanup()

    def test_create_project_missing_name(self, client, app, mock_db_session):
        """422: name is required."""
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            resp = client.post(
                "/api/v1/projects",
                json={"material_source_dir": "/tmp/materials"},
            )
            assert resp.status_code == 422
        finally:
            cleanup()

    def test_create_project_missing_dir(self, client, app, mock_db_session):
        """422: material_source_dir is required."""
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            resp = client.post(
                "/api/v1/projects",
                json={"name": "test"},
            )
            assert resp.status_code == 422
        finally:
            cleanup()


class TestGetProject:
    """GET /api/v1/projects/{project_id}"""

    def test_get_nonexistent_project_returns_404(self, client, app, mock_db_session):
        """404: project does not exist."""
        mock_db_session.set_project_exists(False)
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            resp = client.get("/api/v1/projects/nonexistent-id")
            assert resp.status_code == 404
        finally:
            cleanup()

    def test_get_project_success(self, client, app, mock_db_session):
        """200: existing project is returned."""
        mock_db_session.set_project_exists(True)
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            resp = client.get("/api/v1/projects/proj-001")
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["project_id"] == "proj-001"
            assert "name" in data
            assert "material_source_dir" in data
        finally:
            cleanup()


class TestDeleteProject:
    """DELETE /api/v1/projects/{project_id}"""

    def test_delete_nonexistent_project_returns_404(self, client, app, mock_db_session):
        """404: cannot delete a project that does not exist."""
        mock_db_session.set_project_exists(False)
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            resp = client.delete("/api/v1/projects/nonexistent-id")
            assert resp.status_code == 404
        finally:
            cleanup()

    def test_delete_project_success(self, client, app, mock_db_session):
        """204: successful deletion."""
        mock_db_session.set_project_exists(True)
        cleanup = _override_db_on_all_modules(app, mock_db_session)
        try:
            resp = client.delete("/api/v1/projects/proj-001")
            assert resp.status_code == 204
        finally:
            cleanup()


# ---------------------------------------------------------------------------
# Story / Director
# ---------------------------------------------------------------------------

class TestGenerateStoryboard:
    """POST /api/v1/projects/{project_id}/story"""

    # Use autouse fixture to register director and db overrides
    @pytest.fixture(autouse=True)
    def setup(self, app, mock_db_session, mock_director):
        self._cleanup_db = _override_db_on_all_modules(app, mock_db_session)
        yield
        self._cleanup_db()

    def test_generate_storyboard_success(self, client, mock_db_session):
        """200: storyboard generated successfully."""
        mock_db_session.set_project_exists(True)
        body = {
            "prompt": "旅行素材30秒快剪",
            "style": "快节奏",
            "total_duration_sec": 30,
        }
        resp = client.post("/api/v1/projects/proj-001/story", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["project_id"] == "proj-001"
        assert data["style"] == "快节奏"
        assert len(data["shots"]) == 2
        assert "storyboard_id" in data

    def test_generate_storyboard_project_not_found(self, client, mock_db_session):
        """404: project does not exist."""
        mock_db_session.set_project_exists(False)
        body = {
            "prompt": "test",
            "style": "默认风格",
            "total_duration_sec": 15,
        }
        resp = client.post("/api/v1/projects/nonexistent/story", json=body)
        assert resp.status_code == 404

    def test_generate_storyboard_missing_prompt(self, client, mock_db_session):
        """422: prompt is required."""
        mock_db_session.set_project_exists(True)
        resp = client.post(
            "/api/v1/projects/proj-001/story",
            json={"style": "默认"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

class TestMatchMaterials:
    """POST /api/v1/projects/{project_id}/materials"""

    @pytest.fixture(autouse=True)
    def setup(self, app, mock_db_session, mock_material_agent):
        self._cleanup_db = _override_db_on_all_modules(app, mock_db_session)
        yield
        self._cleanup_db()

    def test_match_materials_success(self, client, mock_db_session):
        """200: materials matched successfully."""
        mock_db_session.set_project_exists(True)
        body = {
            "mode": "retrieval",
            "storyboard": {
                "shots": [
                    {"index": 0, "description": "夕阳下的海滩", "duration_sec": 5.0},
                ]
            },
        }
        resp = client.post("/api/v1/projects/proj-001/materials", json=body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["shot_index"] == 0
        assert data[0]["candidates"][0]["material_id"] == "mat-001"

    def test_match_materials_project_not_found(self, client, mock_db_session):
        """404: project does not exist."""
        mock_db_session.set_project_exists(False)
        body = {
            "mode": "retrieval",
            "storyboard": {"shots": []},
        }
        resp = client.post("/api/v1/projects/nonexistent/materials", json=body)
        assert resp.status_code == 404

    def test_match_materials_invalid_mode(self, client, mock_db_session):
        """400: invalid mode value."""
        mock_db_session.set_project_exists(True)
        body = {
            "mode": "invalid_mode",
            "storyboard": {"shots": []},
        }
        resp = client.post("/api/v1/projects/proj-001/materials", json=body)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------

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
    def setup(self, app, mock_db_session, mock_editor_agent):
        self._cleanup_db = _override_db_on_all_modules(app, mock_db_session)
        yield
        self._cleanup_db()

    def test_start_edit_success(self, client, mock_db_session):
        """200: edit task queued successfully."""
        mock_db_session.set_project_exists(True)
        # Second execute call (task check) should return nothing
        mock_db_session.push_exec_result(None)
        resp = client.post("/api/v1/projects/proj-001/edit", json=self.VALID_BODY)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["task_id"] == "task-abc-123"
        assert data["status"] == "queued"

    def test_start_edit_project_not_found(self, client, mock_db_session):
        """404: project does not exist."""
        mock_db_session.set_project_exists(False)
        resp = client.post("/api/v1/projects/nonexistent/edit", json=self.VALID_BODY)
        assert resp.status_code == 404

    def test_start_edit_invalid_body(self, client, mock_db_session):
        """422: missing required fields."""
        mock_db_session.set_project_exists(True)
        resp = client.post("/api/v1/projects/proj-001/edit", json={})
        assert resp.status_code == 422

    def test_start_edit_without_bgm(self, client, mock_db_session):
        """200: optional bgm_path and subtitles can be omitted."""
        mock_db_session.set_project_exists(True)
        # Second execute call (task check) should return nothing
        mock_db_session.push_exec_result(None)
        body = {
            "storyboard": {"shots": []},
            "material_selections": [],
        }
        resp = client.post("/api/v1/projects/proj-001/edit", json=body)
        assert resp.status_code == 200


class TestGetEditStatus:
    """GET /api/v1/projects/{project_id}/status"""

    def test_get_status_completed(self, client, app, mock_editor_agent):
        """200: completed task status."""
        resp = client.get(
            "/api/v1/projects/proj-001/status",
            params={"task_id": "task-abc-123"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "completed"
        assert data["progress_pct"] == 100

    def test_get_status_in_progress(self, client, app, mock_editor_agent):
        """200: in-progress task status."""
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
    """GET /api/v1/projects/{project_id}/result"""

    def test_get_result_ready(self, client, app, mock_editor_agent):
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

    def test_get_result_not_ready(self, client, app, mock_editor_agent):
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
