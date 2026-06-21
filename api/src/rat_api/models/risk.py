"""Pydantic response models for risk endpoints (T-18)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class RiskFactor(BaseModel):
    feature: str
    contribution: float
    direction: Literal["up", "down"]
    readable: str


class WeekForecast(BaseModel):
    week: date
    risk_score: float = Field(ge=0.0, le=1.0)
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)


class NtaRiskResponse(BaseModel):
    nta_id: str
    current_week: date
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_decile: int = Field(ge=1, le=10)
    top_factors: list[RiskFactor]
    model_version: str
    forecast_12w: list[WeekForecast]


class MapRiskItem(BaseModel):
    nta_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_decile: int = Field(ge=1, le=10)
    nta_name: str | None = None
    centroid_lat: float | None = None
    centroid_lon: float | None = None


class InspectionItem(BaseModel):
    inspection_id: str
    date: date
    result: str
    bbl: str | None
    lat: float | None
    lon: float | None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    db_latency_ms: int
    git_sha: str
