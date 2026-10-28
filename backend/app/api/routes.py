"""HTTP routes for dataset, model, transactions, and graph investigation."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.api import config
from backend.app.api.schemas import ApiResponse, PredictionRequest, TraversalRequest
from backend.app.api.services import ApiServiceError, FraudGuardService
from backend.app.ml_pipeline import DataValidationError

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _service(request: Request) -> FraudGuardService:
    return request.app.state.service


@router.get("/health", response_model=ApiResponse)
def health(request: Request) -> JSONResponse:
    result = _service(request).health()
    return JSONResponse(
        status_code=200 if result["status"] == "ready" else 503,
        content={"data": result},
    )


@router.get("/dataset/summary", response_model=ApiResponse)
def dataset_summary(request: Request) -> dict[str, Any]:
    return {"data": _service(request).dataset_summary()}


@router.get("/dashboard/metrics", response_model=ApiResponse)
def dashboard_metrics(request: Request) -> dict[str, Any]:
    return {"data": _service(request).dashboard_metrics()}


@router.get("/transactions", response_model=ApiResponse)
def transactions(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=config.MAX_PAGE_SIZE),
    transaction_id: str | None = Query(default=None, min_length=1, max_length=128),
    transaction_type: str | None = Query(default=None, alias="type", max_length=32),
    predicted: bool | None = Query(default=None),
    minimum_amount: float | None = Query(default=None, ge=0),
    maximum_amount: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    if minimum_amount is not None and maximum_amount is not None and minimum_amount > maximum_amount:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_amount_range",
                "message": "minimum_amount must not exceed maximum_amount.",
            },
        )
    return {
        "data": _service(request).list_transactions(
            page=page,
            page_size=page_size,
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            predicted=predicted,
            minimum_amount=minimum_amount,
            maximum_amount=maximum_amount,
        )
    }


@router.get("/transactions/{transaction_id}", response_model=ApiResponse)
def transaction_detail(transaction_id: str, request: Request) -> dict[str, Any]:
    result = _service(request).transaction(transaction_id)
    result["connected_accounts"] = [result["sender"], result["recipient"]]
    return {"data": result}


@router.post("/predict", response_model=ApiResponse)
def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
    return {"data": _service(request).predict(payload.model_dump())}


@router.get("/graph", response_model=ApiResponse)
def graph(
    request: Request,
    limit: int = Query(default=500, ge=1, le=config.MAX_GRAPH_TRANSACTIONS),
    transaction_type: str | None = Query(default=None, alias="type", max_length=32),
    predicted: bool | None = Query(default=None),
    minimum_amount: float | None = Query(default=None, ge=0),
    maximum_amount: float | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    if minimum_amount is not None and maximum_amount is not None and minimum_amount > maximum_amount:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_amount_range",
                "message": "minimum_amount must not exceed maximum_amount.",
            },
        )
    return {
        "data": _service(request).graph_data(
            limit=limit,
            transaction_type=transaction_type,
            predicted=predicted,
            minimum_amount=minimum_amount,
            maximum_amount=maximum_amount,
        )
    }


@router.get("/graph/node/{node_id}", response_model=ApiResponse)
def graph_node(
    node_id: str,
    request: Request,
    direction: str = Query(default="both", pattern="^(out|in|both)$"),
    maximum_transactions: int = Query(
        default=500, ge=1, le=config.MAX_GRAPH_TRANSACTIONS
    ),
) -> dict[str, Any]:
    return {
        "data": _service(request).account_graph(
            node_id,
            direction=direction,
            maximum_transactions=maximum_transactions,
        )
    }


@router.post("/graph/traverse", response_model=ApiResponse)
def graph_traverse(payload: TraversalRequest, request: Request) -> dict[str, Any]:
    try:
        result = _service(request).graph_traversal(
            start_node=payload.start_node,
            algorithm=payload.algorithm,
            maximum_depth=payload.maximum_depth,
            direction=payload.direction,
            maximum_nodes=payload.maximum_nodes,
            maximum_edges=payload.maximum_edges,
        )
    except DataValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_graph_request", "message": str(exc)},
        ) from exc
    return {"data": result}


@router.get("/model/summary", response_model=ApiResponse)
def model_summary(request: Request) -> dict[str, Any]:
    return {"data": _service(request).model_summary()}


def install_error_handlers(app: Any) -> None:
    @app.exception_handler(ApiServiceError)
    async def service_error_handler(_request: Request, exc: ApiServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            error = detail
        else:
            error = {
                "code": "http_error",
                "message": str(detail),
            }
        return JSONResponse(status_code=exc.status_code, content={"error": error})

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request fields failed validation.",
                    "details": [
                        {
                            "location": ".".join(str(part) for part in error["loc"]),
                            "message": error["msg"],
                            "type": error["type"],
                        }
                        for error in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unexpected API error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "The request could not be completed.",
                }
            },
        )
