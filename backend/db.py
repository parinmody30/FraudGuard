"""PostgreSQL persistence layer for FraudLab (dataset history).

Reads ``DATABASE_URL`` from the environment (loaded from the project-root ``.env`` by
``backend.app``). The engine is created lazily and the target database is auto-created on
first ``init_db()`` if it does not yet exist, so a fresh Postgres server needs no manual
``CREATE DATABASE``.

If Postgres is unreachable the API degrades gracefully: ``DB_AVAILABLE`` stays ``False`` and
dataset generation still returns its summary (just without persistence / history).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger("fraudlab.db")

# Default points at a local Postgres; override with DATABASE_URL in .env.
DEFAULT_DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/fraudlab"


def _normalize_url(url: str) -> str:
    """Accept plain ``postgresql://`` DSNs and pin them to the installed psycopg2 driver."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    if url.startswith("postgres://"):  # Heroku-style alias
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    return url


DATABASE_URL = _normalize_url(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
DB_AVAILABLE = False


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    return _engine


def _ensure_database_exists() -> None:
    """Create the target database if it is missing by connecting to the server's ``postgres`` db."""
    url = make_url(DATABASE_URL)
    target_db = url.database
    if not target_db:
        return
    admin_url = url.set(database="postgres")
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT", future=True)
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": target_db}
            ).scalar()
            if not exists:
                # Identifier can't be parameterized; target_db comes from our own config.
                conn.execute(text(f'CREATE DATABASE "{target_db}"'))
                logger.info("Created database %s", target_db)
    finally:
        admin_engine.dispose()


def init_db() -> bool:
    """Create the database (if needed) and all tables. Returns True on success."""
    global _SessionLocal, DB_AVAILABLE
    try:
        _ensure_database_exists()
        engine = get_engine()
        # Import models so they are registered on Base.metadata before create_all.
        from backend import models_db  # noqa: F401

        Base.metadata.create_all(engine)
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
        DB_AVAILABLE = True
        logger.info("Database ready at %s", make_url(DATABASE_URL).render_as_string(hide_password=True))
    except Exception as exc:  # noqa: BLE001 — degrade gracefully on any connection/auth error
        DB_AVAILABLE = False
        logger.warning("Database unavailable, dataset history disabled: %s", exc)
    return DB_AVAILABLE


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context manager. Commits on success, rolls back on error."""
    if _SessionLocal is None:
        raise RuntimeError("Database is not initialized")
    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
