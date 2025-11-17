from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import json
import time
import redis.asyncio as redis

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

        # Log request details
        api_logger.info(
            "📥 CHAT REQUEST: query='%s...', user_id='%s', course_id='%s', skip_rag=%s",
            request.query[:100] + "..." if len(request.query) > 100 else request.query,
            request.user_id or "anonymous",
            request.course_id or "global",
            request.skip_rag or False
        )

        # Call chat service for response
        response = await chat_service.chat_completion(request)

        # Log response details
        api_logger.info(
            "📤 CHAT RESPONSE: source='%s', cached=%s, sources_count=%d, latency_ms=%.2f",
            response.response_source,
            response.cached,
            len(response.sources),
            response.latency_ms
        )

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


@router.post("/debug/document-content")
async def debug_document_content() -> Dict[str, Any]:
    """
    Debug endpoint to inspect actual content of stored documents using RedisVL native methods.
    """
    try:
        chat_service = await get_chat_service()

        sample_docs = []

        # Get first few chunk keys using RedisVL approach
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag

        # Use FilterQuery to get sample documents (RedisVL native approach)
        filter_query = FilterQuery(
            return_fields=["text", "material_id", "course_id", "source_file", "chunk_id"],
            num_results=5
        )

        # Execute search
        result = await chat_service.vector_store.index.search(filter_query.query, query_params=filter_query.params)

        api_logger.info(f"Debug: Found {len(result.docs)} documents using RedisVL FilterQuery")

        for i, doc in enumerate(result.docs):
            try:
                # RedisVL automatically handles decoding - no manual binary handling needed!
                sample_docs.append({
                    "doc_number": i + 1,
                    "text": getattr(doc, "text", "NO_TEXT"),
                    "material_id": getattr(doc, "material_id", "NO_MATERIAL_ID"),
                    "course_id": getattr(doc, "course_id", "NO_COURSE_ID"),
                    "source_file": getattr(doc, "source_file", "NO_SOURCE_FILE"),
                    "chunk_id": getattr(doc, "chunk_id", getattr(doc, "payload", "NO_CHUNK_ID")),
                    "text_length": len(getattr(doc, "text", "")),
                    "available_fields": [attr for attr in dir(doc) if not attr.startswith('_') and not callable(getattr(doc, attr, None))]
                })

                api_logger.info(f"Debug: Doc {i+1} - text_length={len(getattr(doc, 'text', ''))}, material_id={getattr(doc, 'material_id', 'None')}")

            except Exception as e:
                api_logger.error(f"Could not process document {i}: {str(e)}")
                sample_docs.append({
                    "doc_number": i + 1,
                    "error": str(e),
                    "text": "ERROR_PROCESSING",
                    "material_id": "ERROR",
                    "course_id": "ERROR",
                    "text_length": 0
                })

        # Get index info using RedisVL
        index_info = await chat_service.vector_store.index.info()

        return {
            "success": True,
            "total_found": len(result.docs),
            "method": "RedisVL FilterQuery (native decoding)",
            "index_info": index_info,
            "sample_documents": sample_docs,
            "diagnosis": {
                "note": "Using RedisVL native methods - no manual binary decoding needed",
                "redisvl_handles": "Automatic field decoding like Redis UI",
                "next_steps": [
                    "If text is still empty, check document ingestion process",
                    "Verify RedisVL schema matches stored data",
                    "Re-ingest documents if needed"
                ]
            }
        }

    except Exception as e:
        api_logger.error(f"❌ Document content debug failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "note": "Failed to use RedisVL native methods"
        }


@router.post("/debug/cache-test")
async def debug_cache_test(
    query: str = "What is ChartInstruct?",
    user_id: str = "test_user",
    course_id: str = "test_course"
) -> Dict[str, Any]:
    """
    Debug endpoint to test cache logic independently.
    """
    try:
        chat_service = await get_chat_service()

        # Log debug info
        api_logger.info(f"🧪 CACHE TEST: Testing query='{query}' for user='{user_id}'")

        # Test storing a response
        api_logger.info("📝 Storing test response in cache...")
        store_success = await chat_service.semantic_cache.store(
            query=query,
            response="This is a test response for debugging cache functionality.",
            user_id=user_id,
            course_id=course_id,
            cache_type="debug_test",
            sources=[{"id": "test_source", "content": "Test source content"}],
            token_usage={"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}
        )

        # Test retrieval
        api_logger.info("🔍 Testing cache retrieval...")
        result = await chat_service.semantic_cache.query(
            query=query,
            user_id=user_id,
            course_id=course_id
        )

        # Get cache stats
        api_logger.info("📊 Getting cache statistics...")
        stats = await chat_service.semantic_cache.get_stats()

        return {
            "success": True,
            "query": query,
            "user_id": user_id,
            "course_id": course_id,
            "store_success": store_success,
            "cache_result": result,
            "cache_stats": stats,
            "distance_threshold": chat_service.semantic_cache.distance_threshold,
            "effective_threshold": min(chat_service.semantic_cache.distance_threshold, 0.5),
            "debug_info": {
                "message": "Cache test completed. Check logs for detailed debugging info.",
                "expected_behavior": "Should see CACHE HIT with reasonable distance < 0.5"
            }
        }

    except Exception as e:
        api_logger.error(f"❌ Cache test failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "message": "Cache test failed - check logs for details"
        }


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

        # Log streaming request details
        api_logger.info(
            "📥 STREAMING REQUEST: query='%s...', user_id='%s', course_id='%s'",
            request.query[:100] + "..." if len(request.query) > 100 else request.query,
            request.user_id or "anonymous",
            request.course_id or "global"
        )

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