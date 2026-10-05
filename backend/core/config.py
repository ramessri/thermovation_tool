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

    # MetricAnything depth fusion — a second, alternative optional densifier
    # (a project may enable LingBot, MetricAnything, both, or neither; each
    # writes its own artifact). Runs in the main worker env: its requirements
    # pin torch<2.5.0, which the worker's torch 2.4.1 satisfies — no venv.
    ENABLE_METRICANYTHING_FUSION: bool = False
    METRICANYTHING_VENDOR_DIR: Path = Path("/opt/metric-anything")
    METRICANYTHING_CHECKPOINT_REPO: str = "yjh001/metricanything_student_depthmap"
    METRICANYTHING_CHECKPOINT_FILE: str = "student_depthmap.pt"
    METRICANYTHING_CACHE_DIR: Path = Path("/app/models/metricanything")
    # One forward pass per frame (no windowed mode) — cap and evenly subsample.
    METRICANYTHING_MAX_FRAMES: int = 200

    # HVAC wall-mount placement (indoor_room only) — recommends where to mount a
    # unit, anchored on the Rücklauf (return pipe). Gated two-tier: this switch
    # AND the per-project hvac_mode flag. GDINO + SegFormer run in the main
    # worker env via transformers; SAM2 has its own kill switch with a
    # box-centroid fallback.
    ENABLE_HVAC_PLACEMENT: bool = False
    HVAC_GDINO_MODEL_ID: str = "IDEA-Research/grounding-dino-base"
    HVAC_ENABLE_SAM2: bool = True
    HVAC_SAM2_MODEL_ID: str = "facebook/sam2.1-hiera-small"
    HVAC_SEGFORMER_MODEL_ID: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    HVAC_MIN_WALL_INLIERS: int = 2000
    HVAC_MIN_CLEARANCE_CM: float = 45.0
    # A wall candidate whose best frame has less than this ADE20K wall-pixel
    # fraction is excluded (a bed's mattress edge once passed the inlier floor
    # and the 25° tilt gate at 3% wall pixels). None = never checked = allowed.
    HVAC_MIN_ADE20K_CONFIDENCE: float = 0.15
    HVAC_UNIT_SIZE_CM: str = "60x40"        # candidate mounting rectangle, W x H
    # GDINO+SAM2 per-frame cost is high — cap and evenly subsample above this.
    HVAC_MAX_DETECTION_FRAMES: int = 150
    # Frames sampled for ADE20K wall segmentation (only points an image
    # classified as "wall" reach the plane fit).
    HVAC_MAX_WALL_SEG_FRAMES: int = 40


settings = Settings()
