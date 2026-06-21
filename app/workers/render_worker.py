import os
import logging
from celery import Task
from app.core.celery_app import celery_app
from app.services.render_service import RenderService

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="render_video")
def render_video(
    self,
    project_id: str,
    storyboard: dict,
    material_selections: list[dict],
    bgm_path=None,
) -> dict:
    """异步渲染视频任务"""
    try:
        logger.info(f"[render_video] START project={project_id} shots={len(storyboard.get('shots',[]))} mats={len(material_selections)}")

        if self.request.id:
            self.update_state(state="PROCESSING", meta={"progress_pct": 10})

        # 构建时间线
        service = RenderService()
        logger.info("[render_video] Building timeline...")
        clip = service.build_timeline(storyboard, material_selections, bgm_path)

        if self.request.id:
            self.update_state(state="RENDERING", meta={"progress_pct": 50})

        # 渲染
        import uuid
        task_id = self.request.id or str(uuid.uuid4())[:8]
        output_filename = f"{project_id}_{task_id}.mp4"
        logger.info(f"[render_video] Rendering to {output_filename}...")
        output_path = service.render(clip, output_filename)

        # 添加 BGM
        if bgm_path and os.path.exists(str(bgm_path)):
            if self.request.id:
                self.update_state(state="RENDERING", meta={"progress_pct": 80})
            logger.info("[render_video] Adding BGM...")
            output_path = service.add_bgm(output_path, bgm_path)

        # 清理
        clip.close()

        logger.info(f"[render_video] DONE: {output_path}")
        return {
            "status": "completed",
            "output_path": output_path,
            "progress_pct": 100,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception(f"Render failed for project {project_id}")
        logger.error(f"Full traceback:\n{tb}")
        return {
            "status": "failed",
            "error": str(e),
            "traceback": tb,
            "progress_pct": 0,
        }
