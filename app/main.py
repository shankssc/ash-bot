# app/main.py
"""
AniRAG: Production-grade anime intelligence microservice.
FastAPI application entry point with observability and safety patterns.
"""

import sys

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router

# Initialize logging BEFORE any other imports that might log
# This handles config validation errors safely
from app.core.config import settings
from app.core.exceptions import (
    ConfigurationError,
    ServiceUnavailableError,
    handle_configuration_error,
    handle_service_unavailable,
)
from app.core.logging import get_logger, init_logging

# Initialize logging with settings
init_logging(settings)
logger = get_logger(__name__)

# Log startup context BEFORE importing heavy dependencies
logger.info(
    f"Starting {settings.APP_NAME} v{settings.APP_VERSION} "
    f"in {settings.ENVIRONMENT} environment"
)
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Working directory: {Path.cwd()}")

# Now safe to import FastAPI and routers


def create_app() -> FastAPI:
    """Factory function for FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Anime intelligence RAG microservice built on 100% free infrastructure",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
    )

    # ===== MIDDLEWARE =====
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ===== EXCEPTION HANDLERS =====
    app.add_exception_handler(ServiceUnavailableError, handle_service_unavailable)
    app.add_exception_handler(ConfigurationError, handle_configuration_error)

    # ===== ROUTERS =====
    app.include_router(api_router)

    # ===== STARTUP EVENTS =====
    @app.on_event("startup")
    async def startup_event():
        logger.info("🚀 Application startup complete")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Server: http://{settings.HOST}:{settings.PORT}")
        logger.info(f"CORS origins: {settings.cors_origins_list}")

        # Log deprecation warning if ChromaDB path is set
        if settings.VECTOR_DB_PATH:
            logger.warning(
                "⚠️  DEPRECATION WARNING: ChromaDB (VECTOR_DB_PATH) is configured. "
                "Migrating to Qdrant Cloud per AniRAG spec. This path will be ignored."
            )

    # ===== SHUTDOWN EVENTS =====
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("🛑 Application shutdown initiated")

    # ===== ROOT ENDPOINT =====
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            content={
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "status": "running",
                "docs": "/docs" if settings.ENVIRONMENT != "production" else None,
                "health": "/api/v1/health",
            }
        )

    return app


# Global app instance (for uvicorn)
app = create_app()

if __name__ == "__main__":
    import uvicorn  # type: ignore

    logger.info(f"Starting uvicorn server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        workers=settings.WORKERS if not settings.is_development() else 1,
    )
