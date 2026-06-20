"""FastAPI dependency injection utilities."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.db import async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get a database session."""
    async with async_session_maker() as session:
        yield session
