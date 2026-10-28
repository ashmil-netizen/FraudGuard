"""Typed HTTP request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PredictionRequest(StrictModel):
    step: int = Field(ge=1)
    amount: float = Field(ge=0)
    oldbalanceOrg: float | None = None
    newbalanceOrig: float | None = None
    oldbalanceDest: float | None = None
    newbalanceDest: float | None = None
    type: str = Field(min_length=1, max_length=32)

    @field_validator("type")
    @classmethod
    def normalize_transaction_type(cls, value: str) -> str:
        normalized = value.strip()
        return normalized or "__missing__"


class TraversalRequest(StrictModel):
    start_node: str = Field(min_length=1, max_length=128)
    algorithm: Literal["bfs", "dfs"]
    maximum_depth: int = Field(default=2, ge=0, le=6)
    direction: Literal["out", "in", "both"] = "both"
    maximum_nodes: int = Field(default=500, ge=1, le=2_000)
    maximum_edges: int = Field(default=5_000, ge=1, le=10_000)


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    data: dict[str, Any]
