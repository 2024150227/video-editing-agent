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
        with patch("app.agents.material.IndexService") as MockIndex:
            mock_index = MagicMock()
            MockIndex.return_value = mock_index

            agent = MaterialAgent()
            results = agent.match_materials_generation(storyboard)

        assert len(results) == 2  # 两个 shot 各一组结果
