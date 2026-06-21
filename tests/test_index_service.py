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
