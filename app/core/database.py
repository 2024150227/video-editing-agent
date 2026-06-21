from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


# Lazy initialization to allow alembic to import models without asyncpg
_engine = None
_async_session = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
    return _engine


def get_async_session():
    global _async_session
    if _async_session is None:
        _async_session = async_sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _async_session


async def get_db():
    session_maker = get_async_session()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
