"""
Storage abstraction used by the pipeline and API to move files between local
scratch space and the configured backend.

Objects are addressed by a relative key, e.g. "<project_id>/sfm/cameras.json".
Only STORAGE_BACKEND=local is implemented; the other modes listed in
.env.example (synology_webdav, synology_s3, filestack) raise NotImplementedError.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from backend.core.config import settings


class LocalStorageBackend:
    """Stores objects as files under settings.LOCAL_STORAGE_ROOT."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self.root / key

    async def upload(self, local_path: Path, key: str) -> str:
        dest = self._path_for(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(local_path).resolve() != dest.resolve():
            shutil.copyfile(local_path, dest)
        return key

    async def download(self, key: str, local_path: Path) -> Path:
        src = self._path_for(key)
        if not src.exists():
            raise FileNotFoundError(f"storage key not found: {key} (looked at {src})")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != local_path.resolve():
            shutil.copyfile(src, local_path)
        return local_path

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()


_storage_instance = None


def get_storage():
    global _storage_instance
    if _storage_instance is None:
        if settings.STORAGE_BACKEND != "local":
            raise NotImplementedError(
                f"STORAGE_BACKEND={settings.STORAGE_BACKEND!r} is not implemented — only 'local' is available."
            )
        _storage_instance = LocalStorageBackend(settings.LOCAL_STORAGE_ROOT)
    return _storage_instance
