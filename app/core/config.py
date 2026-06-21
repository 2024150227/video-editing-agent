from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 应用
    APP_NAME: str = "Video Editing Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # 数据库
    DATABASE_URL: str = "postgresql+asyncpg://vagent:vagent_secret@localhost:5432/vagent"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://vagent:vagent_secret@localhost:5432/vagent"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "materials"

    # 火山方舟
    VOLCANO_ARK_API_KEY: str = ""
    VOLCANO_ARK_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    VOLCANO_ARK_MODEL: str = "doubao-vision-pro-32k"

    # CLIP
    CLIP_MODEL: str = "ViT-B/32"

    # 视频
    DEFAULT_RESOLUTION: str = "1080x1920"
    DEFAULT_ASPECT_RATIO: str = "9:16"
    DEFAULT_FPS: int = 30
    OUTPUT_DIR: str = "./output"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # 并发控制
    MAX_CONCURRENT_RENDERS: int = 1

    # 中文字体路径 (用于字幕)
    FONT_PATH: str = "/usr/share/fonts/NotoSansSC-Regular.ttf"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
