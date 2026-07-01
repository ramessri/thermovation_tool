from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # App
    APP_ENV: str = Field(default="development")
    SECRET_KEY: str = Field(default="change_me_in_production")

    @model_validator(mode="after")
    def _validate_production_secrets(self) -> "Settings":
        if self.APP_ENV == "production":
            if self.SECRET_KEY == "change_me_in_production":
                raise ValueError("SECRET_KEY must be set to a secure random value in production")
            if len(self.SECRET_KEY) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters in production")
        return self
    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:3001")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://photogram:photogram@postgres:5432/photogram"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    # Storage
    STORAGE_BACKEND: str = "local"
    LOCAL_STORAGE_ROOT: Path = Path("/app/storage")

    # Synology WebDAV
    SYNOLOGY_WEBDAV_URL: str = ""
    SYNOLOGY_USER: str = ""
    SYNOLOGY_PASSWORD: str = ""

    # S3-compatible / MinIO
    S3_ENDPOINT_URL: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_BUCKET: str = "photogram"

    # Filestack
    FILESTACK_API_KEY: str = ""

    # ML
    MODEL_CACHE_DIR: Path = Path("./models")

    # Pipeline
    TEMP_DIR: Path = Path("/tmp/photogram")
    MAX_FRAMES_PER_VIDEO: int = 500
    FRAME_EXTRACTION_FPS: float = 2.0

    # LingBot-Map depth fusion (optional densification stage — runs in an
    # isolated torch-2.8 venv via subprocess; see CLAUDE.md). Off by default.
    ENABLE_LINGBOT_FUSION: bool = False
    LINGBOT_VENV_PYTHON: str = "/opt/lingbot-venv/bin/python"
    LINGBOT_INFER_SCRIPT: str = "/app/backend/workers/pipeline/lingbot/infer_depth.py"
    LINGBOT_CHECKPOINT: Path = Path("/app/models/lingbot/lingbot-map.pt")
    LINGBOT_CHECKPOINT_REPO: str = "robbyant/lingbot-map"
    LINGBOT_CHECKPOINT_FILE: str = "lingbot-map.pt"
    # Memory knobs proven on a 12 GB GPU (RTX 4070 Ti).
    LINGBOT_NUM_SCALE_FRAMES: int = 2
    LINGBOT_CAMERA_ITERS: int = 1
    LINGBOT_WINDOWED_THRESHOLD: int = 60    # >this uses bounded windowed mode (12 GB-safe; streaming KV peaks ~64 frames)
    LINGBOT_WINDOW_SIZE: int = 24
    LINGBOT_CONF_PERCENTILE: float = 40.0
    # Cap frames fed to depth inference — long walkthroughs (1000+ frames) OOM
    # even in windowed mode, and indoor coverage is highly redundant. Evenly
    # subsample above this. ~300 views is ample for densification on a 12 GB GPU.
    LINGBOT_MAX_FRAMES: int = 300


settings = Settings()
