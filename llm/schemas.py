from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high", "unknown"]


class URLAnalysisRequest(BaseModel):
    id: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=8192)


class AnalysisResponse(BaseModel):
    id: str
    risk_level: RiskLevel
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    summary: str
    reasons: list[str]
    details: dict[str, Any]
