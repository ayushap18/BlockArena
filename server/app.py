"""
FastAPI application for the BlockArena environment.

Exposes the environment through OpenEnv-compatible HTTP/WebSocket endpoints.
"""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies and ensure the OpenEnv package is available."
    ) from e

try:
    from models import BlockArenaAction, BlockArenaObservation
    from server.blockarena_environment import BlockArenaEnvironment
except ImportError:
    from ..models import BlockArenaAction, BlockArenaObservation
    from .blockarena_environment import BlockArenaEnvironment


app = create_app(
    BlockArenaEnvironment,
    BlockArenaAction,
    BlockArenaObservation,
    env_name="blockarena",
    max_concurrent_envs=1,
)


# Add root route and health check
@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "BlockArena",
        "description": "Strategic Contract Negotiation Environment for OpenEnv",
        "team": "Codecatalysts - OpenEnv Hackathon 2026",
        "version": "1.0.0",
        "endpoints": {
            "reset": "POST /reset - Start a new negotiation episode",
            "step": "POST /step - Take an action in the current episode",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation (Swagger UI)",
            "openapi": "GET /openapi.json - OpenAPI schema"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "blockarena",
        "ready": True
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()