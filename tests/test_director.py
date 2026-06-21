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
                    "mood_words": ["自由", "启程"],
                }
            ],
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
            mock_client.chat_json = AsyncMock(
                side_effect=[
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
                        ],
                    },
                ]
            )
            agent = DirectorAgent()
            result = await agent.generate_storyboard("test", "快节奏", 30)
            assert sum(s["duration_sec"] for s in result["shots"]) == 30
