"""Tests for app/api/materials.py — Materials API with LangGraph."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.main import create_app
from app.api.deps import get_graph


def _mock_graph_for_materials(material_matches):
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "material_matches": material_matches,
        "current_step": "materials_approved",
    })
    return mock_graph


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestMatchMaterials:
    @pytest.fixture(autouse=True)
    def setup(self, app):
        self.matches = [
            {"shot_index": 1, "candidates": [
                {"material_id": "m1", "file_name": "v1.mp4", "clip_range": [0,5], "score": 0.92, "reason": "match"},
            ]},
        ]

        # Graph override
        graph = _mock_graph_for_materials(self.matches)
        app.dependency_overrides[get_graph] = lambda: graph

        # DB override — session as async generator function
        from app.api import materials as mod
        db_session = AsyncMock()
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = MagicMock(id="proj-001")
        db_session.execute = AsyncMock(return_value=mock_exec)

        async def _db_override():
            yield db_session

        app.dependency_overrides[mod.get_db] = _db_override
        self._db = db_session
        yield
        app.dependency_overrides.clear()

    def test_materials_success(self, client):
        resp = client.post("/api/v1/projects/proj-001/materials", json={
            "mode": "retrieval",
            "storyboard": {"shots": [{"index": 1, "description": "test", "duration_sec": 5}]},
        })
        assert resp.status_code == 200
        assert resp.json()[0]["candidates"][0]["score"] == 0.92

    def test_materials_404(self, client):
        self._db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        resp = client.post("/api/v1/projects/nope/materials", json={
            "mode": "retrieval", "storyboard": {"shots": []},
        })
        assert resp.status_code == 404

    def test_materials_bad_mode(self, client):
        resp = client.post("/api/v1/projects/proj-001/materials", json={
            "mode": "bad", "storyboard": {"shots": []},
        })
        assert resp.status_code == 400

    def test_materials_generation(self, client):
        resp = client.post("/api/v1/projects/proj-001/materials", json={
            "mode": "generation",
            "storyboard": {"shots": [{"index": 1, "description": "test", "duration_sec": 5}]},
        })
        assert resp.status_code == 200
