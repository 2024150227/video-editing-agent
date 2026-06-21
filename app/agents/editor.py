import logging
import os
from celery.result import AsyncResult
from app.workers.render_worker import render_video

logger = logging.getLogger(__name__)


class EditorAgent:
    """剪辑 Agent — 提交渲染任务、查询状态"""

    def submit_edit(
        self,
        project_id: str,
        storyboard: dict,
        material_selections: list[dict],
        bgm_path: str | None = None,
    ) -> str:
        """提交异步剪辑任务，返回 task_id"""
        task = render_video.delay(
            project_id=project_id,
            storyboard=storyboard,
            material_selections=material_selections,
            bgm_path=bgm_path,
        )
        logger.info(f"Edit task submitted: {task.id} for project {project_id}")
        return task.id

    def get_task_status(self, task_id: str) -> dict:
        """查询渲染任务状态"""
        result = AsyncResult(task_id, app=render_video.app)

        response = {
            "task_id": task_id,
            "status": result.state,
            "progress_pct": 0,
            "output_path": None,
            "error": None,
        }

        if result.ready():
            if result.successful():
                data = result.result
                response["status"] = "completed"
                response["progress_pct"] = 100
                response["output_path"] = data.get("output_path") if isinstance(data, dict) else None
            else:
                response["status"] = "failed"
                response["error"] = str(result.info) if result.info else "Unknown error"
        elif result.state == "PROGRESS":
            meta = result.info or {}
            response["progress_pct"] = meta.get("progress_pct", 0)
        elif result.state == "STARTED":
            response["progress_pct"] = 5

        return response

    def get_output_url(self, task_id: str) -> dict:
        """获取渲染完成的视频访问信息"""
        status = self.get_task_status(task_id)
        if status["status"] != "completed":
            return {"ready": False, "status": status["status"]}

        output_path = status.get("output_path")
        if output_path and os.path.exists(output_path):
            return {
                "ready": True,
                "file_path": output_path,
                "file_size_bytes": os.path.getsize(output_path),
            }
        return {"ready": False, "error": "Output file not found"}
