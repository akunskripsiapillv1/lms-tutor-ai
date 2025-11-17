"""
Main FastAPI application entry point.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.core.logger import app_logger

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.exceptions import TutorServiceException
from app.services.optimized_chat_service import OptimizedChatService

# Global service instances
_chat_service: OptimizedChatService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    app_logger.info("Starting Tutor Services API")

    try:
        # Initialize optimized chat service
        global _chat_service
        _chat_service = OptimizedChatService()
        await _chat_service.connect()

        # Run health checks
        health = await _chat_service.health_check()
        if not all(health.values()):
            app_logger.warning("Some components may not be fully healthy: health=%s", health)

        app_logger.info("Tutor Services API started successfully")
        app_logger.info("API available at http://%s:%s", settings.api_host, settings.api_port)
        app_logger.info("Health check at http://%s:%s/health", settings.api_host, settings.api_port)
        app_logger.info("API docs at http://%s:%s/docs", settings.api_host, settings.api_port)

        yield

    except Exception as e:
        app_logger.error("Failed to start Tutor Services API: %s", str(e))
        raise
    finally:
        # Shutdown
        app_logger.info("Shutting down Tutor Services API")

        try:
            if _chat_service:
                await _chat_service.disconnect()
        except Exception as e:
            app_logger.error("Error during shutdown: %s", str(e))

        app_logger.info("Tutor Services API shutdown complete")


# Create FastAPI application with lifespan
app = FastAPI(
    title="Tutor Services API",
    description="RAG Chatbot with Semantic Caching - MVP Implementation",
    version="1.0.0",
    debug=settings.api_debug,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(health_router)


@app.exception_handler(TutorServiceException)
async def tutor_service_exception_handler(request, exc: TutorServiceException):
    """Handle custom TutorServiceException."""
    app_logger.error(
        "Tutor service error",
        error_code=getattr(exc, 'error_code', None),
        message=exc.message,
        details=getattr(exc, 'details', None),
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": getattr(exc, 'error_code', 'TUTOR_SERVICE_ERROR'),
            "message": exc.message,
            "details": getattr(exc, 'details', None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    app_logger.error(
        "Unhandled exception",
        error=str(exc),
        exc_info=True,
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred"
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Tutor Services API",
        "version": "1.0.0",
        "status": "healthy",
        "description": "RAG Chatbot with Semantic Caching",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "features": [
            "Chat completion with RAG",
            "Semantic caching",
            "Streaming responses",
            "User personalization",
            "Analytics and monitoring"
        ]
    }


# Development server configuration
if __name__ == "__main__":
    app_logger.info("Starting Tutor Services API in development mode")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
        access_log=True
    )