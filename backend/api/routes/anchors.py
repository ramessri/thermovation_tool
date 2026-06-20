"""Anchor wizard endpoints — upload reference object, specify dimensions."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.api.deps import get_db
from backend.core.storage import get_storage
from backend.core.config import settings
from backend.models.models import Anchor, Project

router = APIRouter()


class AnchorDimensions(BaseModel):
    name: str
    width_mm: float
    height_mm: float
    depth_mm: float | None = None


class AnchorResponse(BaseModel):
    id: str
    name: str
    width_mm: float | None
    height_mm: float | None
    depth_mm: float | None

    class Config:
        from_attributes = True


@router.post("/{project_id}/anchors")
async def upload_anchor(
    project_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a reference photo of the anchor object."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(404, "Project not found")

    storage = get_storage()
    dest_key = f"{project_id}/anchors/{file.filename}"

    tmp_path = settings.TEMP_DIR / f"anchor_{project_id}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        await storage.upload(tmp_path, dest_key)
    finally:
        tmp_path.unlink(missing_ok=True)

    anchor = Anchor(
        project_id=project_id,
        name=file.filename,
        storage_key=dest_key,
    )
    db.add(anchor)
    await db.commit()
    await db.refresh(anchor)

    return anchor


@router.post("/{project_id}/anchors/{anchor_id}/dimensions")
async def set_dimensions(
    project_id: str,
    anchor_id: str,
    body: AnchorDimensions,
    db: AsyncSession = Depends(get_db),
):
    """Set real-world dimensions for the anchor object."""
    result = await db.execute(select(Anchor).where(Anchor.id == anchor_id))
    anchor = result.scalar_one_or_none()
    if not anchor:
        raise HTTPException(404, "Anchor not found")

    anchor.name = body.name
    anchor.width_mm = body.width_mm
    anchor.height_mm = body.height_mm
    anchor.depth_mm = body.depth_mm
    await db.commit()
    await db.refresh(anchor)

    return anchor


@router.get("/{project_id}/anchors/{anchor_id}", response_model=AnchorResponse)
async def get_anchor(
    project_id: str,
    anchor_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Anchor).where(Anchor.id == anchor_id))
    anchor = result.scalar_one_or_none()
    if not anchor:
        raise HTTPException(404, "Anchor not found")
    return anchor
