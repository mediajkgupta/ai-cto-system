"""
app.py — FastAPI application factory for AI CTO System.

Usage:
    uvicorn ai_cto.api.app:app --reload

Or import `create_app()` in tests to get a fresh app instance.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ai_cto.api.routes.projects import router as projects_router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load .env on startup so ANTHROPIC_API_KEY is available in all threads."""
    load_dotenv()
    yield


def create_app() -> FastAPI:
    """Factory function — returns a configured FastAPI application."""
    app = FastAPI(
        title="AI CTO System API",
        description=(
            "Turn a plain-language idea into working software. "
            "The pipeline runs Planning → Coding → Verification → Execution, "
            "with automatic debugging and memory persistence."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )

    app.include_router(projects_router)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    return app


# Module-level app instance — used by uvicorn and test client
app = create_app()
