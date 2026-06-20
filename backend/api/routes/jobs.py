"""Job status endpoints."""
from fastapi import APIRouter, HTTPException
from backend.workers.tasks import celery_app

router = APIRouter()


@router.get("/{task_id}")
async def get_job_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status":  result.status,
        "result":  result.result if result.ready() else None,
    }
