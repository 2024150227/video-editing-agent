"""Test video render service — uses mock MoviePy to avoid slow video encoding."""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from app.services.render_service import RenderService


@pytest.fixture
def render_service(settings_override):
    """Return a RenderService instance (settings_override sets OUTPUT_DIR)."""
    with patch("app.services.render_service.get_settings") as mock_get:
        mock_get.return_value = settings_override
        svc = RenderService()
    return svc


class TestRenderServiceBuildTimeline:

    def test_build_timeline_basic(self, render_service):
        """Should build a CompositeVideoClip from storyboard + material selections."""
        storyboard = {
            "shots": [
                {"index": 0, "description": "a cat walking", "duration_sec": 5},
                {"index": 1, "description": "a dog barking", "duration_sec": 3},
            ]
        }
        selections = [
            {"shot_index": 0, "file_path": "/fake/cat.mp4", "clip_range": [0.0, 5.0]},
            {"shot_index": 1, "file_path": "/fake/dog.mp4", "clip_range": [0.0, 3.0]},
        ]

        with patch("app.services.render_service.VideoFileClip") as mock_vfc, \
                patch("app.services.render_service.concatenate_videoclips") as mock_concat, \
                patch("os.path.exists", return_value=True):
            mock_clip_a = MagicMock()
            mock_clip_a.duration = 10.0
            mock_clip_b = MagicMock()
            mock_clip_b.duration = 10.0
            mock_vfc.side_effect = [mock_clip_a, mock_clip_b]

            mock_concat.return_value = MagicMock()

            result = render_service.build_timeline(storyboard, selections)

            assert mock_concat.called
            assert result is mock_concat.return_value

    def test_build_timeline_skips_missing_material(self, render_service):
        """Should skip shots with no matching material selection."""
        storyboard = {
            "shots": [
                {"index": 0, "description": "a cat", "duration_sec": 5},
            ]
        }
        selections = []  # no matching selection

        with patch("app.services.render_service.VideoFileClip"), \
                patch("app.services.render_service.concatenate_videoclips"):
            with pytest.raises(ValueError, match="No valid clips"):
                render_service.build_timeline(storyboard, selections)

    def test_build_timeline_skips_missing_file(self, render_service):
        """Should skip a shot whose file_path does not exist (os.path.exists=False)."""
        storyboard = {
            "shots": [
                {"index": 0, "description": "a cat", "duration_sec": 5},
            ]
        }
        selections = [
            {"shot_index": 0, "file_path": "/nonexistent/video.mp4", "clip_range": [0.0, 5.0]},
        ]

        with patch("app.services.render_service.VideoFileClip"), \
                patch("app.services.render_service.concatenate_videoclips"), \
                patch("os.path.exists", return_value=False):
            with pytest.raises(ValueError, match="No valid clips"):
                render_service.build_timeline(storyboard, selections)

    def test_build_timeline_image_support(self, render_service):
        """Should handle image files with ImageClip."""
        storyboard = {
            "shots": [
                {"index": 0, "description": "a landscape photo", "duration_sec": 5},
            ]
        }
        selections = [
            {"shot_index": 0, "file_path": "/fake/photo.png", "clip_range": [0.0, 5.0]},
        ]

        with patch("app.services.render_service.ImageClip") as mock_ic, \
                patch("app.services.render_service.VideoFileClip") as mock_vfc, \
                patch("app.services.render_service.concatenate_videoclips") as mock_concat, \
                patch("os.path.exists", return_value=True):
            mock_img = MagicMock()
            mock_ic.return_value = mock_img
            mock_img.duration = 10.0
            mock_concat.return_value = MagicMock()

            result = render_service.build_timeline(storyboard, selections)

            # Should use ImageClip, not VideoFileClip
            assert mock_ic.called
            assert not mock_vfc.called

    def test_build_timeline_resizes_to_target(self, render_service):
        """Each clip should be resized to the configured resolution."""
        storyboard = {
            "shots": [
                {"index": 0, "description": "cat", "duration_sec": 3},
            ]
        }
        selections = [
            {"shot_index": 0, "file_path": "/fake/cat.mp4", "clip_range": [0.0, 3.0]},
        ]

        with patch("app.services.render_service.VideoFileClip") as mock_vfc, \
                patch("app.services.render_service.concatenate_videoclips") as mock_concat, \
                patch("os.path.exists", return_value=True):
            # _load_clip calls clip.subclipped() and returns a different mock;
            # ensure the chain (with_duration -> resized) works on that result.
            mock_sub = MagicMock()
            mock_sub.with_duration.return_value = mock_sub
            mock_sub.resized.return_value = mock_sub

            mock_clip = MagicMock()
            mock_clip.duration = 10.0
            mock_clip.subclipped.return_value = mock_sub
            mock_vfc.return_value = mock_clip
            mock_concat.return_value = MagicMock()

            render_service.build_timeline(storyboard, selections)

            # Should resize; default resolution is 1080x1920
            mock_sub.resized.assert_called_once_with(new_size=(1080, 1920))

    def test_build_timeline_subclips(self, render_service):
        """Should subclip video when clip_range differs from full duration."""
        storyboard = {
            "shots": [
                {"index": 0, "description": "cat", "duration_sec": 3},
            ]
        }
        selections = [
            {"shot_index": 0, "file_path": "/fake/cat.mp4", "clip_range": [1.0, 4.0]},
        ]

        with patch("app.services.render_service.VideoFileClip") as mock_vfc, \
                patch("app.services.render_service.concatenate_videoclips") as mock_concat, \
                patch("os.path.exists", return_value=True):
            mock_sub = MagicMock()
            mock_sub.with_duration.return_value = mock_sub
            mock_sub.resized.return_value = mock_sub

            mock_clip = MagicMock()
            mock_clip.duration = 10.0
            mock_clip.subclipped.return_value = mock_sub
            mock_vfc.return_value = mock_clip
            mock_concat.return_value = MagicMock()

            render_service.build_timeline(storyboard, selections)

            # Should have called subclipped
            mock_clip.subclipped.assert_called_once()

    def test_build_timeline_with_bgm(self, render_service):
        """bgm_path parameter is accepted (actual BGM added in worker)."""
        storyboard = {
            "shots": [
                {"index": 0, "description": "cat", "duration_sec": 3},
            ]
        }
        selections = [
            {"shot_index": 0, "file_path": "/fake/cat.mp4", "clip_range": [0.0, 3.0]},
        ]

        with patch("app.services.render_service.VideoFileClip") as mock_vfc, \
                patch("app.services.render_service.concatenate_videoclips") as mock_concat, \
                patch("os.path.exists", return_value=True):
            mock_sub = MagicMock()
            mock_sub.with_duration.return_value = mock_sub
            mock_sub.resized.return_value = mock_sub

            mock_clip = MagicMock()
            mock_clip.duration = 10.0
            mock_clip.subclipped.return_value = mock_sub
            mock_vfc.return_value = mock_clip
            mock_concat.return_value = MagicMock()

            # bgm_path should not cause any error
            result = render_service.build_timeline(
                storyboard, selections, bgm_path="/fake/bgm.mp3"
            )
            assert result is mock_concat.return_value


class TestRenderServiceRender:

    def test_render_writes_file(self, render_service):
        """Render should call write_videofile and return output path."""
        mock_clip = MagicMock()
        output_filename = "test_output.mp4"

        with patch("os.makedirs"):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.getsize", return_value=5 * 1024 * 1024):
                    result = render_service.render(mock_clip, output_filename)

            assert mock_clip.write_videofile.called
            assert result.endswith(output_filename)

    def test_render_validates_output(self, render_service):
        """Should raise FileNotFoundError if output file doesn't appear."""
        mock_clip = MagicMock()
        output_filename = "missing.mp4"

        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Render output not found"):
                render_service.render(mock_clip, output_filename)


class TestRenderServiceAddBGM:

    def test_add_bgm_runs_ffmpeg(self, render_service):
        """add_bgm should call ffmpeg via subprocess."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = render_service.add_bgm("/fake/video.mp4", "/fake/bgm.mp3")
            assert mock_run.called
            assert "_bgm.mp4" in result

    def test_add_bgm_raises_on_ffmpeg_fail(self, render_service):
        """add_bgm should raise OSError if ffmpeg fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("ffmpeg not found")
            with pytest.raises(OSError, match="ffmpeg not found"):
                render_service.add_bgm("/fake/video.mp4", "/fake/bgm.mp3")


class TestRenderServiceInternal:

    def test_find_material_found(self, render_service):
        """_find_material should return matching selection by shot_index."""
        selections = [
            {"shot_index": 0, "file_path": "/a.mp4"},
            {"shot_index": 1, "file_path": "/b.mp4"},
        ]
        result = render_service._find_material(1, selections)
        assert result == {"shot_index": 1, "file_path": "/b.mp4"}

    def test_find_material_not_found(self, render_service):
        """_find_material should return None when no match."""
        selections = [{"shot_index": 0, "file_path": "/a.mp4"}]
        result = render_service._find_material(99, selections)
        assert result is None

    def test_validate_output_valid(self, render_service):
        """_validate_output should return True for an existing file."""
        with patch("os.path.exists", return_value=True), \
                patch("os.path.getsize", return_value=1024 * 1024):
            assert render_service._validate_output("/fake/video.mp4") is True

    def test_validate_output_missing(self, render_service):
        """_validate_output should raise for a non-existent file."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Render output not found"):
                render_service._validate_output("/fake/missing.mp4")


class TestRenderWorker:
    """Test the Celery render_video task.

    Because ``render_video`` is a module-level ``PromiseProxy`` whose
    underlying Task object shares mutable ``request`` state across calls,
    this class uses a **per-test fixture** to save/restore the request id.
    """

    @pytest.fixture(autouse=True)
    def _isolate_task_request(self):
        """Reset Celery Task shared state before each test.

        ``render_video._render_service`` caches a service instance from the
        previous test.  If we don't reset it, the second test picks up the
        previous test's mock, bypassing any new patches.
        """
        from app.workers.render_worker import render_video

        task = render_video._get_current_object()
        task._render_service = None
        orig_id = getattr(task.request, "id", None)
        yield
        task._render_service = None
        if orig_id is None:
            task.request.id = None
        else:
            task.request.id = orig_id

    def test_render_video_task_success(self):
        """Verify the render_video Celery task runs end-to-end."""
        from app.workers.render_worker import render_video
        task = render_video._get_current_object()

        storyboard = {
            "shots": [
                {"index": 0, "description": "cat", "duration_sec": 3},
            ]
        }
        selections = [{"shot_index": 0, "file_path": "/fake/cat.mp4"}]

        with patch.object(task, "update_state"), \
                patch("app.workers.render_worker.RenderService") as MockService:
            mock_svc = MagicMock()
            MockService.return_value = mock_svc

            mock_clip = MagicMock()
            mock_svc.build_timeline.return_value = mock_clip
            mock_svc.render.return_value = "/output/test_xxx.mp4"

            task.request.id = "celery-task-id-123"

            with patch("os.path.exists", return_value=False):
                result = task.run("proj-1", storyboard, selections, bgm_path=None)

            assert result["status"] == "completed"
            assert result["output_path"] == "/output/test_xxx.mp4"
            mock_svc.build_timeline.assert_called_once_with(
                storyboard, selections, None
            )
            mock_svc.render.assert_called_once()

    def test_render_video_task_with_bgm(self):
        """Should add BGM when bgm_path is provided and exists."""
        from app.workers.render_worker import render_video
        task = render_video._get_current_object()

        storyboard = {
            "shots": [{"index": 0, "description": "cat", "duration_sec": 3}]
        }
        selections = [{"shot_index": 0, "file_path": "/fake/cat.mp4"}]

        with patch.object(task, "update_state"), \
                patch("app.workers.render_worker.RenderService") as MockService:
            mock_svc = MagicMock()
            MockService.return_value = mock_svc

            mock_clip = MagicMock()
            mock_svc.build_timeline.return_value = mock_clip
            mock_svc.render.return_value = "/output/test.mp4"
            mock_svc.add_bgm.return_value = "/output/test_bgm.mp4"

            task.request.id = "celery-task-id-456"

            with patch("os.path.exists", return_value=True):
                result = task.run(
                    "proj-1", storyboard, selections, bgm_path="/fake/bgm.mp3"
                )

            assert result["status"] == "completed"
            assert result["output_path"] == "/output/test_bgm.mp4"
            mock_svc.add_bgm.assert_called_once()

    def test_render_video_task_failure(self):
        """Should return failed status on exception."""
        from app.workers.render_worker import render_video
        task = render_video._get_current_object()

        storyboard = {
            "shots": [{"index": 0, "description": "cat", "duration_sec": 3}]
        }
        selections = [{"shot_index": 0, "file_path": "/fake/cat.mp4"}]

        with patch.object(task, "update_state"), \
                patch("app.workers.render_worker.RenderService") as MockService:
            mock_svc = MagicMock()
            MockService.return_value = mock_svc
            mock_svc.build_timeline.side_effect = ValueError("No valid clips")

            task.request.id = "celery-task-id-789"

            with patch("os.path.exists", return_value=False):
                result = task.run(
                    "proj-1", storyboard, selections, bgm_path=None
                )

            assert result["status"] == "failed"
            assert "No valid clips" in result["error"]
