import os
import logging

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}


class FileService:
    """素材文件扫描服务"""

    @staticmethod
    def scan_directory(dir_path: str) -> list[dict]:
        """递归扫描目录，返回所有支持的媒体文件"""
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory not found: {dir_path}")

        files = []
        for root, _, filenames in os.walk(dir_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                all_exts = SUPPORTED_VIDEO | SUPPORTED_IMAGE | SUPPORTED_AUDIO
                if ext in all_exts:
                    full_path = os.path.join(root, fname)
                    file_type = (
                        "video" if ext in SUPPORTED_VIDEO
                        else "image" if ext in SUPPORTED_IMAGE
                        else "audio"
                    )
                    files.append({
                        "file_name": fname,
                        "file_path": full_path,
                        "file_type": file_type,
                        "ext": ext,
                    })
        logger.info(f"Scanned {dir_path}: found {len(files)} media files")
        return files

    @staticmethod
    def get_video_duration(file_path: str) -> float:
        """获取视频时长（秒），依赖 FFmpeg"""
        try:
            import subprocess
            import json
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            info = json.loads(result.stdout)
            return float(info["format"].get("duration", 0))
        except Exception as e:
            logger.warning(f"Failed to get duration for {file_path}: {e}")
            return 0.0

    @staticmethod
    def extract_keyframe(file_path: str, output_path: str, at_second: float = 1.0) -> bool:
        """从视频中提取关键帧图片"""
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-ss", str(at_second), "-i", file_path,
                "-vframes", "1", "-q:v", "2", output_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            logger.warning(f"Failed to extract keyframe from {file_path}: {e}")
            return False
