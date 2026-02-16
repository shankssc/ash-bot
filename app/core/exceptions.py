# app/core/exceptions.py
"""
Custom exceptions and handlers for consistent error responses.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse


class ServiceUnavailableError(Exception):
    """Raised when a critical external service is unavailable."""

    pass


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""

    pass


async def handle_service_unavailable(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "status": "service_unavailable"},
    )


async def handle_configuration_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "status": "configuration_error"},
    )
