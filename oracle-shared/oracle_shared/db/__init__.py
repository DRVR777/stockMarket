"""oracle_shared.db — async PostgreSQL layer for ORACLE long-term storage.

Usage::

    from oracle_shared.db import get_engine, get_session, init_db

    engine = get_engine()          # uses DATABASE_URL env var
    async with get_session() as s: # async context manager
        ...
    await init_db(engine)          # CREATE TABLE IF NOT EXISTS
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from oracle_shared.db.models import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

DEFAULT_DATABASE_URL = "postgresql+asyncpg://oracle:oracle@localhost:5432/oracle"


def get_engine(url: str | None = None) -> AsyncEngine:
    """Return (and cache) the global async engine."""
    global _engine
    if _engine is None:
        db_url = url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        _engine = create_async_engine(
            db_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session that auto-commits on success, rolls back on error."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db(engine: AsyncEngine | None = None) -> None:
    """Create all tables if they don't exist."""
    eng = engine or get_engine()
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose the engine connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
