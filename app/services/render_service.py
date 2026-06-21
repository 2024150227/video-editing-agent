import os
import subprocess
import logging
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RenderService:
    """视频渲染服务 — 时间线构建 + 合成输出"""

    def __init__(self):
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    def build_timeline(
        self,
        storyboard: dict,
        material_selections: list[dict],
        bgm_path: str | None = None,
    ) -> CompositeVideoClip:
        """根据分镜和素材构建时间线"""
        clips = []
        target_w, target_h = map(int, settings.DEFAULT_RESOLUTION.split("x"))

        for shot in storyboard["shots"]:
            shot_index = shot["index"]
            material = self._find_material(shot_index, material_selections)
            if not material:
                logger.warning(f"No material found for shot {shot_index}, skipping")
                continue

            clip = self._load_clip(material, shot)
            if clip is None:
                continue

            # 裁剪到指定时长
            duration = shot["duration_sec"]
            clip = clip.with_duration(duration)

            # 缩放到目标分辨率
            clip = clip.resized(new_size=(target_w, target_h))

            # 转场处理（当前版本：直接拼接，转场效果后续迭代）
            clips.append(clip)

        if not clips:
            raise ValueError("No valid clips to compose")

        return concatenate_videoclips(clips, method="compose")

    def _find_material(self, shot_index: int, selections: list[dict]) -> dict | None:
        for s in selections:
            if s.get("shot_index") == shot_index:
                return s
        return None

    def _load_clip(self, material: dict, shot: dict):
        """加载并裁剪素材片段"""
        file_path = material.get("file_path") or material.get("file_name")
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None

        clip_range = material.get("clip_range", [0, shot["duration_sec"]])
        start, end = clip_range[0], clip_range[1]

        if file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            clip = ImageClip(file_path)
            clip = clip.with_duration(end - start)
        else:
            clip = VideoFileClip(file_path)
            if start > 0 or end < clip.duration:
                clip = clip.subclipped(start, min(end, clip.duration))

        return clip

    def render(self, clip: CompositeVideoClip, output_filename: str) -> str:
        """渲染输出 MP4 文件"""
        output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

        clip.write_videofile(
            output_path,
            fps=settings.DEFAULT_FPS,
            codec="libx264",
            audio_codec="aac",
            temp_audio_file_path=settings.OUTPUT_DIR,
            threads=4,
            preset="medium",
        )

        self._validate_output(output_path)
        return output_path

    @staticmethod
    def _validate_output(file_path: str) -> bool:
        """校验输出文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Render output not found: {file_path}")
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Render complete: {file_path} ({size_mb:.1f} MB)")
        return True

    def add_bgm(self, video_path: str, bgm_path: str, volume: float = 0.3) -> str:
        """为视频添加背景音乐（使用 FFmpeg）"""
        output_path = video_path.replace(".mp4", "_bgm.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", bgm_path,
            "-filter_complex", f"[1:a]volume={volume}[a1];[0:a][a1]amix=inputs=2:duration=first",
            "-c:v", "copy",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
