from .simple_chat import router as chat_router
from .health import router as health_router
from .websocket_chat import router as websocket_router

__all__ = [
    "chat_router",
    "health_router",
    "websocket_router"
]