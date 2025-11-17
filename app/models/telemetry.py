from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class TelemetryEvent(BaseModel):
    """Model for telemetry events."""
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    duration_ms: Optional[float] = Field(None, description="Event duration in milliseconds")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TelemetryMetrics(BaseModel):
    """Model for aggregated telemetry metrics."""
    total_requests: int = Field(default=0, description="Total number of requests")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate percentage")
    average_latency_ms: float = Field(default=0.0, description="Average response latency")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    total_tokens: int = Field(default=0, description="Total tokens used")
    unique_users: int = Field(default=0, description="Number of unique users")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    time_window_hours: int = Field(..., description="Time window for metrics in hours")


class CostTracker(BaseModel):
    """Model for cost tracking."""
    prompt_tokens: int = Field(default=0, description="Number of prompt tokens")
    completion_tokens: int = Field(default=0, description="Number of completion tokens")
    total_tokens: int = Field(default=0, description="Total tokens")
    cost_usd: float = Field(default=0.0, description="Total cost in USD")
    model: str = Field(..., description="Model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
