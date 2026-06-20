"""
WebSocket endpoint — relays pipeline progress from Redis pub/sub to browser.
Connect at: ws://localhost:8000/ws/projects/{project_id}/progress
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis

from backend.core.config import settings

router = APIRouter()


@router.websocket("/projects/{project_id}/progress")
async def project_progress(websocket: WebSocket, project_id: str):
    await websocket.accept()

    r = aioredis.from_url(settings.REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"project:{project_id}:progress")

    # Replay last known state immediately so the client isn't blank on connect.
    last = await r.get(f"project:{project_id}:last_progress")
    if last:
        await websocket.send_text(last.decode())

    async def relay():
        """Forward Redis pub/sub messages to the WebSocket."""
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"].decode())

    async def wait_disconnect():
        """Return when the client closes the connection."""
        try:
            while True:
                await websocket.receive()
        except (WebSocketDisconnect, Exception):
            return

    relay_task = asyncio.create_task(relay())
    disconnect_task = asyncio.create_task(wait_disconnect())
    try:
        done, pending = await asyncio.wait(
            {relay_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        await pubsub.unsubscribe()
        try:
            await r.aclose()
        except AttributeError:
            r.close()
