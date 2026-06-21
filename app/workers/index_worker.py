import os
import tempfile
import logging
from celery import Task
from app.core.celery_app import celery_app
from app.services.file_service import FileService
from app.services.index_service import IndexService
from app.models.project import IndexStatus

logger = logging.getLogger(__name__)


class IndexTask(Task):
    _index_service = None

    @property
    def index_service(self):
        if self._index_service is None:
            self._index_service = IndexService()
        return self._index_service


@celery_app.task(bind=True, base=IndexTask, name="index_project")
def index_project(self, project_id: str, dir_path: str) -> dict:
    """异步索引项目素材目录"""
    try:
        files = FileService.scan_directory(dir_path)
        indexed = 0
        errors = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            for file_info in files:
                self.update_state(
                    state="PROGRESS",
                    meta={"indexed": indexed, "total": len(files)},
                )

                try:
                    keyframe_path = None
                    if file_info["file_type"] == "video":
                        keyframe_path = os.path.join(
                            tmpdir, f"{file_info['file_name']}_kf.jpg"
                        )
                        if FileService.extract_keyframe(file_info["file_path"], keyframe_path, at_second=1.0):
                            self.index_service.index_single(project_id, file_info, keyframe_path)
                            indexed += 1

                    elif file_info["file_type"] == "image":
                        self.index_service.index_single(project_id, file_info, file_info["file_path"])
                        indexed += 1

                except Exception as e:
                    logger.error(f"Failed to index {file_info['file_path']}: {e}")
                    errors += 1
                    continue

        return {
            "status": "completed",
            "indexed": indexed,
            "total": len(files),
            "errors": errors,
        }

    except Exception as e:
        logger.exception(f"Indexing failed for project {project_id}")
        return {"status": "failed", "error": str(e)}
