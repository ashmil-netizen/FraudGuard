"""FraudGuard AI FastAPI entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import config
from backend.app.api.routes import install_error_handlers, router
from backend.app.api.services import FraudGuardService


def create_app(service: FraudGuardService | None = None) -> FastAPI:
    api_service = service or FraudGuardService()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        api_service.initialize()
        yield

    app = FastAPI(
        title="FraudGuard AI API",
        description=(
            "PaySim test-split transaction search, saved-model inference, and "
            "bounded account graph investigation. Model scores are not calibrated "
            "fraud probabilities; graph traversal does not classify fraud."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.service = api_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    install_error_handlers(app)
    return app


app = create_app()
