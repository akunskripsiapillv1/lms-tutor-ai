"""
Simple Chat Service - Clean Orchestration Layer
Single responsibility: Check cache → If miss, use RAG → Cache response
No duplicate logic - each service handles its own responsibilities
"""
from typing import Dict, Any, Optional, AsyncGenerator
from .unified_rag_service import UnifiedRAGService, lcel_rag_service
from .custom_cache_service import CustomCacheService
from ..core.logger import chat_logger
from ..config.settings import settings

class SimpleChatService:
    """
    Clean chat service that orchestrates cache and RAG without duplicate logic
    Responsibilities:
    - Check cache first
    - If cache miss: use RAG
    - Cache the RAG response
    """

    def __init__(self):
        """Initialize chat service with RAG and Cache"""
        self.rag_service = UnifiedRAGService()
        self.cache_service = CustomCacheService()
        chat_logger.info("SimpleChatService initialized - clean orchestration")

    async def chat(self, query: str, user_id: Optional[str] = None, course_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main chat method - clean orchestration without duplicate logic

        Args:
            query: User query
            user_id: Optional user ID
            course_id: Optional course ID for context filtering

        Returns:
            Complete response with metadata from either cache or RAG
        """
        try:
            # Step 1: Query cache service (following reference notebook pattern)
            # cache_service.query() handles both hit/miss and personalization internally
            cached_result = await self.cache_service.query(query, user_id, course_id)

            if cached_result is not None:
                # Cache hit (could be raw or personalized)
                chat_logger.info(f"Cache hit for user={user_id}, course={course_id}")
                return {
                    "response": cached_result,
                    "source": "cache",
                    "cached": True,
                    "sources": [],  # Cache responses don't have RAG sources by design
                    "user_id": user_id,
                    "course_id": course_id
                }

            # Step 2: Cache miss - use RAG (RAG service handles document retrieval)
            rag_response = await self.rag_service.query(query, course_id)

            # Step 3: Store the RAG response in cache for future use
            if rag_response.get("answer"):
                # Generate embedding and store response using reference pattern
                embedding = await self.cache_service.generate_embedding(query)
                await self.cache_service.store_response(
                    prompt=query,
                    response=rag_response["answer"],
                    embedding=embedding,
                    user_id=user_id or "anonymous",
                    model=settings.openai_model,
                    course_id=course_id
                )

            chat_logger.info(f"RAG response for user={user_id}, course={course_id}")
            return {
                "response": rag_response["answer"],
                "source": "rag",
                "cached": False,
                "sources": rag_response.get("sources", []),
                "user_id": user_id,
                "course_id": course_id
            }

        except Exception as e:
            chat_logger.error(f"Chat request failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "source": "error",
                "cached": False,
                "sources": [],
                "user_id": user_id,
                "course_id": course_id,
                "error": str(e)
            }


# =================================================================
# LCEL PATTERN IMPLEMENTATION
# =================================================================
# LangChain Expression Language pattern - separate orchestration layer
# =================================================================

    async def chat_lcel(self, query: str, user_id: Optional[str] = None,
                       course_id: Optional[str] = None, rag_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        LCEL chat method - separate orchestrator using LangChain patterns.
        Same cache logic, different RAG implementation.

        Args:
            query: User query
            user_id: Optional user ID
            course_id: Optional course ID for context filtering
            rag_threshold: Custom RAG threshold (overrides settings.rag_distance_threshold)

        Returns:
            Complete response with metadata from LCEL processing
        """
        try:
            # Step 1: Cache check (using same cache service!)
            cached_result = await self.cache_service.query(query, user_id, course_id)

            if cached_result is not None:
                # Cache hit (same logic as original chat)
                chat_logger.info(f"LCEL Cache hit for user={user_id}, course={course_id}")
                return {
                    "response": cached_result,
                    "source": "cache",
                    "cached": True,
                    "sources": [],
                    "user_id": user_id,
                    "course_id": course_id,
                    "method": "lcel_cached"
                }

            # Step 2: Cache miss - Use LCEL RAG service (different RAG implementation)
            rag_response = await lcel_rag_service.query(
                question=query,
                course_id=course_id,
                rag_threshold=rag_threshold or settings.rag_distance_threshold,
                top_k=settings.rag_top_k
            )

            # Step 3: Cache the LCEL response (same caching logic!)
            if rag_response.get("answer"):
                embedding = await self.cache_service.generate_embedding(query)
                await self.cache_service.store_response(
                    prompt=query,
                    response=rag_response["answer"],
                    embedding=embedding,
                    user_id=user_id or "anonymous",
                    model=settings.openai_model,
                    course_id=course_id
                )

            chat_logger.info(f"LCEL RAG response for user={user_id}, course={course_id}")
            return {
                "response": rag_response["answer"],
                "source": "rag",
                "cached": False,
                "sources": rag_response.get("sources", []),
                "user_id": user_id,
                "course_id": course_id,
                "context_quality": rag_response.get("context_quality", "unknown"),
                "rag_threshold": rag_response.get("rag_threshold", settings.rag_distance_threshold),
                "response_time_ms": rag_response.get("response_time_ms", 0),
                "method": "lcel_rag"
            }

        except Exception as e:
            chat_logger.error(f"LCEL Chat request failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request with LCEL.",
                "source": "error",
                "cached": False,
                "sources": [],
                "user_id": user_id,
                "course_id": course_id,
                "error": str(e),
                "method": "lcel_error"
            }

    async def chat_stream_lcel(self, query: str, user_id: Optional[str] = None,
                              course_id: Optional[str] = None, rag_threshold: Optional[float] = None):
        """
        LCEL streaming chat - orchestrator for streaming responses.
        Cache check first, then LCEL streaming.

        Yields:
            Response chunks as they're generated
        """
        try:
            # Cache check first (same logic!)
            cached_result = await self.cache_service.query(query, user_id, course_id)

            if cached_result is not None:
                yield cached_result
                chat_logger.info(f"LCEL Streaming cache hit for user={user_id}")
                return

            # Cache miss: Use LCEL RAG streaming
            async for chunk in lcel_rag_service.stream(
                question=query,
                course_id=course_id,
                rag_threshold=rag_threshold or settings.rag_distance_threshold,
                top_k=settings.rag_top_k
            ):
                if chunk:
                    yield chunk

            chat_logger.info(f"LCEL Streaming completed for user={user_id}, course={course_id}")

        except Exception as e:
            chat_logger.error(f"LCEL Streaming chat failed: {e}")
            yield f"Error: {str(e)}"