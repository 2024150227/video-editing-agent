import os
import logging
from celery import Task
from app.core.celery_app import celery_app
from app.services.render_service import RenderService
from app.models.task import TaskStatus

logger = logging.getLogger(__name__)


class RenderTask(Task):
    """自定义 Celery Task，支持进度回调和状态更新"""
    _render_service = None

    @property
    def render_service(self):
        if self._render_service is None:
            self._render_service = RenderService()
        return self._render_service


@celery_app.task(bind=True, base=RenderTask, name="render_video")
def render_video(
    self,
    project_id: str,
    storyboard: dict,
    material_selections: list[dict],
    bgm_path: str | None = None,
) -> dict:
    """异步渲染视频任务"""
    try:
        self.update_state(state="PROCESSING", meta={"progress_pct": 10})

        # 构建时间线
        service = self.render_service
        clip = service.build_timeline(storyboard, material_selections, bgm_path)

        self.update_state(state="RENDERING", meta={"progress_pct": 50})

        # 渲染
        output_filename = f"{project_id}_{self.request.id}.mp4"
        output_path = service.render(clip, output_filename)

        # 添加 BGM
        if bgm_path and os.path.exists(bgm_path):
            self.update_state(state="RENDERING", meta={"progress_pct": 80})
            output_path = service.add_bgm(output_path, bgm_path)

        # 清理
        clip.close()

        return {
            "status": "completed",
            "output_path": output_path,
            "progress_pct": 100,
        }

    except Exception as e:
        logger.exception(f"Render failed for project {project_id}")
        return {
            "status": "failed",
            "error": str(e),
            "progress_pct": 0,
        }
