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
            mock_result.ready.return_value = False
            mock_result.state = "PROGRESS"
            mock_result.info = {"progress_pct": 65}
            MockResult.return_value = mock_result

            agent = EditorAgent()
            status = agent.get_task_status("task_001")
            assert status["status"] == "PROGRESS"
            assert status["progress_pct"] == 65
