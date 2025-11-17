from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime

from ..services.optimized_chat_service import OptimizedChatService
from ..core.logger import api_logger
router = APIRouter(prefix="/health", tags=["health"])

# Global service instance (simple for MVP)
_chat_service: Optional[OptimizedChatService] = None


async def get_chat_service() -> OptimizedChatService:
    """Dependency to get chat service instance."""
    global _chat_service
    if _chat_service is None:
        try:
            _chat_service = OptimizedChatService()
            await _chat_service.connect()
        except Exception as e:
            api_logger.error("Failed to initialize chat service: %s", str(e))
            raise HTTPException(status_code=503, detail="Service unavailable")
    return _chat_service


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    """
    try:
        # Simple health check without full component testing
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Tutor Services API",
            "version": "1.0.0"
        }
    except Exception as e:
        api_logger.error("Health check failed: %s", str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/components")
async def component_health_check() -> Dict[str, Any]:
    """
    Detailed health check for all components.
    """
    try:
        component_status = {
            "chat_service": False,
            "redis": False,
            "vector_index": False,
            "llm_client": False,
            "embedding_service": False
        }

        # Test chat service (includes all dependencies)
        try:
            chat_service = await get_chat_service()
            health = await chat_service.health_check()

            component_status.update({
                "chat_service": True,
                "redis": health.get("redis", False),
                "vector_index": health.get("vector_index", False),
                "llm_client": health.get("llm_client", False),
                "embedding_service": health.get("embedding_service", False)
            })
        except Exception as e:
            api_logger.error("Component health check failed: %s", str(e))

        # Overall status
        overall_healthy = all(component_status.values())

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": component_status,
            "overall_healthy": overall_healthy
        }

    except Exception as e:
        api_logger.error("Component health check failed: %s", str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/metrics")
async def get_health_metrics() -> Dict[str, Any]:
    """
    Get basic health and performance metrics.
    """
    try:
        # Get service statistics if available
        chat_service = await get_chat_service()
        analytics = await chat_service.get_chat_analytics()  # Overall analytics

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "service_metrics": {
                "active_connections": 1,  # Simple placeholder
                "total_requests": analytics.get("total_requests", 0),
                "cache_hit_rate": analytics.get("cache_stats", {}).get("hit_rate_percent", 0),
                "average_latency_ms": analytics.get("average_latency_ms", 0)
            },
            "component_metrics": {
                "health_check_count": 1,
                "last_health_check": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        api_logger.error("Failed to get health metrics: %s", str(e))
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "service_metrics": {},
            "component_metrics": {}
        }