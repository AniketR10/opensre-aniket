"""Deployment environment selection.

A leaf with no ``config`` imports: anything that varies by environment (Clerk
instance, Tracer base URL, Sentry environment tag) reads it from here without
pulling in the subject it belongs to.
"""

import os
from enum import StrEnum

__all__ = (
    "Environment",
    "get_environment",
)


class Environment(StrEnum):
    """Application environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_environment() -> Environment:
    """Get current environment from ENV variable.

    Returns:
        Environment enum value based on ENV variable.
        Defaults to DEVELOPMENT if not set or unrecognized.
    """
    env_value = os.getenv("ENV", "development").lower()
    if env_value in ("production", "prod"):
        return Environment.PRODUCTION
    return Environment.DEVELOPMENT
