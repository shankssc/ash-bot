"""
Health check endpoint with dependency validation.
"""

from fastapi import APIRouter, Depends, status

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.core.secrets import secrets

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)


def _check_qdrant(settings: Settings) -> dict[str, str]:
    """Validate Qdrant configuration."""
    try:
        if not settings.QDRANT_URL or "your-uuid" in settings.QDRANT_URL:
            return {"status": "degraded", "message": "QDRANT_URL not configured"}
        if not secrets.QDRANT_API_KEY.get_secret_value():
            return {"status": "degraded", "message": "QDRANT_API_KEY missing"}
        return {"status": "ok", "url": settings.QDRANT_URL}
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return {"status": "error", "message": str(e)}


def _check_groq(secrets_obj) -> dict[str, str]:
    """Validate Groq configuration."""
    try:
        if not secrets_obj.GROQ_API_KEY.get_secret_value():
            return {"status": "degraded", "message": "GROQ_API_KEY missing"}
        return {"status": "ok", "model": "llama-3.1-70b-versatile"}
    except Exception as e:
        logger.warning(f"Groq health check failed: {e}")
        return {"status": "error", "message": str(e)}


def _check_redis(secrets_obj) -> dict[str, str]:
    """Validate Redis configuration."""
    try:
        redis_url = secrets_obj.REDIS_URL.get_secret_value()
        if not redis_url or "upstash" not in redis_url.lower():
            return {"status": "degraded", "message": "REDIS_URL not configured"}
        return {"status": "ok", "url": "[REDACTED]"}
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return {"status": "error", "message": str(e)}


@router.get(
    "",
    summary="Service health check",
    responses={
        200: {"description": "All dependencies healthy"},
        503: {"description": "One or more dependencies degraded/unavailable"},
    },
)
async def health_check(settings: Settings = Depends(get_settings)):
    """
    Comprehensive health check of all external dependencies.

    Returns 200 if all critical dependencies are healthy.
    Returns 503 if any critical dependency is degraded/unavailable.
    """
    checks = {
        "app": {
            "status": "ok",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        },
        "qdrant": _check_qdrant(settings),
        "groq": _check_groq(secrets),
        "redis": _check_redis(secrets),
    }

    # Determine overall status: error > degraded > ok
    statuses = [v["status"] for v in checks.values() if isinstance(v, dict)]
    if "error" in statuses:
        overall_status = "error"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif "degraded" in statuses:
        overall_status = "degraded"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        overall_status = "ok"
        status_code = status.HTTP_200_OK

    if status_code != 200:
        logger.warning(f"Health check degraded: {checks}")

    return {
        "status": overall_status,
        "timestamp": settings.APP_VERSION,  # Will be replaced with real timestamp later
        "checks": checks,
    }
