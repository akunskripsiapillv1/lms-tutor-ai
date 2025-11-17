from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import json
import time

from ..models.chat import ChatRequest, ChatResponse
from ..services.optimized_chat_service import OptimizedChatService
from ..core.exceptions import TutorServiceException
from ..core.logger import api_logger
router = APIRouter(prefix="/chat", tags=["chat"])

# Global service instance (simple for MVP)
_chat_service: Optional[OptimizedChatService] = None


async def get_chat_service() -> OptimizedChatService:
    """Dependency to get chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = OptimizedChatService()
        await _chat_service.connect()
    return _chat_service


@router.post("/completion", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    chat_service: OptimizedChatService = Depends(get_chat_service)
) -> ChatResponse:
    """
    Generate chat completion response with RAG and semantic caching.
    """
    try:
        # Simple validation
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 10000:
            raise HTTPException(status_code=400, detail="Query too long (max 10000 characters)")

        # Call chat service for response
        response = await chat_service.chat_completion(request)

        # Schedule background tasks (telemetry)
        background_tasks.add_task(
            track_chat_request,
            request.query,
            request.user_id,
            response.cached,
            response.latency_ms
        )

        return response

    except TutorServiceException as e:
        api_logger.error("Chat completion failed: error=%s, details=%s", e.message, getattr(e, 'details', None))
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        api_logger.error("Unexpected error in chat completion: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/completion/stream")
async def chat_completion_stream(
    request: ChatRequest,
    chat_service: OptimizedChatService = Depends(get_chat_service)
) -> StreamingResponse:
    """
    Generate streaming chat completion response.
    """
    request_start = time.time()

    try:
        # Simple validation
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 10000:
            raise HTTPException(status_code=400, detail="Query too long (max 10000 characters)")

        async def generate():
            """Generate streaming response."""
            try:
                async for chunk in chat_service.chat_completion_stream(request):
                    # Check if chunk contains token usage metadata
                    if "[TOKEN_USAGE]" in chunk and "[/TOKEN_USAGE]" in chunk:
                        # Extract and send token usage as structured data
                        try:
                            start_idx = chunk.find("[TOKEN_USAGE]") + len("[TOKEN_USAGE]")
                            end_idx = chunk.find("[/TOKEN_USAGE]")
                            token_usage_str = chunk[start_idx:end_idx]
                            token_usage = json.loads(token_usage_str)

                            # Send token usage as metadata
                            yield f"event: token_usage\n"
                            yield f"data: {json.dumps(token_usage)}\n\n"
                        except json.JSONDecodeError as e:
                            api_logger.warning("Failed to parse token usage metadata: %s", str(e))
                            # Send the raw chunk if parsing fails
                            yield f"data: {chunk}\n\n"
                    else:
                        # Regular content chunk
                        yield f"data: {chunk}\n\n"

                yield "data: [DONE]\n\n"

                # Log streaming request tracking at the end
                latency_ms = (time.time() - request_start) * 1000
                api_logger.info(
                    "Streaming chat request tracked: query_length=%d, user_id=%s, cached=streaming, latency_ms=%.2f",
                    len(request.query),
                    request.user_id or "anonymous",
                    latency_ms
                )

            except Exception as e:
                api_logger.error("Streaming error: %s", str(e))
                yield f"event: error\n"
                yield f"data: Streaming error: {str(e)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )

    except TutorServiceException as e:
        api_logger.error("Streaming chat completion failed: %s", e.message)
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        api_logger.error("Unexpected error in streaming chat completion: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")



async def track_chat_request(query: str, user_id: Optional[str], cached: bool, latency_ms: float):
    """
    Background task to track chat request metrics.
    """
    try:
        api_logger.info(
            "Chat request tracked: query_length=%s, user_id=%s, cached=%s, latency_ms=%s",
            len(query), user_id or "anonymous", cached, latency_ms
        )
        # TODO: Add to metrics storage when implemented
    except Exception as e:
        api_logger.error("Failed to track chat request: %s", str(e))