"""FastAPI application factory for the FraudLab backend (Phase 7.0 / 7.2).

Assembles the router stubs into a single app and enables permissive CORS for local frontend
development. Run with:  uvicorn backend.app:app --reload

On import this loads the project-root ``.env`` (Phase 7.2) so ``GROQ_API_KEY`` is available to
the Investigation Agent at request time. When the key is absent or invalid the agent
transparently uses its deterministic offline fallback — the response contract is unchanged.
"""

from __future__ import annotations

import os

# Load environment variables from the project-root .env as early as possible, before any
# component reads os.getenv("GROQ_API_KEY"). Resolved relative to this file so it works
# regardless of the process working directory. Existing real env vars are NOT overridden.
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.api import ALL_ROUTERS  # noqa: E402
from backend.api.dependencies import API_VERSION  # noqa: E402


def create_app() -> FastAPI:
    app = FastAPI(
        title="FraudLab API",
        version=API_VERSION,
        description="Backend contract for the FraudLab fraud-intelligence frontend.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten for production
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in ALL_ROUTERS:
        app.include_router(router)
    return app


app = create_app()
