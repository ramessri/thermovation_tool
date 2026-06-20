"""Database session factory and utilities."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.core.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    """Get a database session."""
    async with async_session_maker() as session:
        yield session
