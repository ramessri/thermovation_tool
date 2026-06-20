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


settings = Settings()
