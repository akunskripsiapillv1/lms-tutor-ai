"""
Optimized Chat Service

This replaces the complex ChatService with a cleaner implementation
that integrates simplified semantic cache and RAG services.
"""
from typing import Dict, Any, Optional, List, AsyncGenerator
import time

from ..config.settings import settings
from ..models.chat import ChatRequest, ChatResponse
from ..core.llm_client import LLMClient
from ..core.telemetry import TelemetryService
from ..core.exceptions import TutorServiceException
from ..services.simplified_semantic_cache import SimplifiedSemanticCache
from ..services.simplified_rag_service import SimplifiedRAGService
from ..repositories.unified_vector_store import unified_vector_store
from ..core.logger import api_logger


class OptimizedChatService:
    """
    Optimized chat service that cleanly separates concerns:

    1. Semantic Cache: For query deduplication and performance optimization
    2. RAG Service: For knowledge-based responses using document retrieval
    3. LLM Client: For generating responses when cache/RAG miss
    4. Unified Vector Store: Handles all storage operations

    Replaces: ChatService (complex implementation with mixed concerns)
    """

    def __init__(self):
        """Initialize optimized chat service with clean dependency injection."""
        # Core components
        self.llm_client = LLMClient()
        self.telemetry_service = TelemetryService()
        self.telemetry = self.telemetry_service.get_telemetry_logger()

        # Business logic services
        self.semantic_cache = SimplifiedSemanticCache(
            llm_client=self.llm_client,
            telemetry=self.telemetry
        )
        self.rag_service = SimplifiedRAGService(
            llm_client=self.llm_client,
            telemetry=self.telemetry
        )

        # Storage layer
        self.vector_store = unified_vector_store

        self._connected = False

        api_logger.info("OptimizedChatService initialized")

    async def connect(self) -> None:
        """Connect to all required services."""
        try:
            # Connect to unified vector store (handles both Redis instances)
            await self.vector_store.connect()

            self._connected = True
            api_logger.info("OptimizedChatService connected successfully")

        except Exception as e:
            api_logger.error("Failed to connect OptimizedChatService: %s", str(e))
            raise TutorServiceException(f"Connection failed: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from all services."""
        try:
            await self.vector_store.disconnect()
            self._connected = False
            api_logger.info("OptimizedChatService disconnected")

        except Exception as e:
            api_logger.error("Error during disconnect: %s", str(e))

    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Generate chat completion with RAG + Cache + Personalization flow.

        Flow: RAG → Cache Check → [Hit/Personalize] or [Miss → Generate] → Response
        """
        if not self._connected:
            await self.connect()

        start_time = time.time()

        try:
            # STEP 0: Always try RAG enhancement first (knowledge base retrieval)
            rag_chunks = []
            if not request.skip_rag:
                rag_chunks = await self.rag_service.retrieve_documents(
                    query=request.query,
                    material_ids=None
                )

            # STEP 1: Check semantic cache
            cache_result = await self.semantic_cache.query(
                query=request.query,
                user_id=request.user_id,
                course_id=request.course_id
            )

            if cache_result["cached"] and cache_result["response"]:
                # CACHE HIT scenarios
                if request.context:
                    # Scenario: Cache Hit + Personalization (hit_personalized)
                    personalization_result = await self.semantic_cache.personalize_response(
                        cached_response=cache_result["response"],
                        user_context=request.context,
                        original_query=request.query,
                        user_id=request.user_id
                    )

                    if personalization_result["personalized"]:
                        response_text = personalization_result["response"]
                        if isinstance(response_text, list):
                            response_text = str(response_text)
                        elif not isinstance(response_text, str):
                            response_text = str(response_text)

                        # Log comprehensive telemetry for personalized cache hit
                        self.telemetry.log(
                            user_id=request.user_id,
                            method="chat_completion",
                            cache_status="hit_personalized",
                            response_source="cache_personalized",
                            latency_ms=personalization_result.get("latency_ms", 0),
                            input_tokens=personalization_result.get("input_tokens", 0),
                            output_tokens=personalization_result.get("output_tokens", 0),
                            metadata={
                                "course_id": request.course_id,
                                "rag_available": bool(rag_chunks),
                                "rag_used": False,
                                "cache_hit": True,
                                "personalization_used": True,
                                "model": personalization_result.get("model", "gpt-4o-mini"),
                                "cost_usd": personalization_result.get("cost_usd", 0),
                                "savings_usd": self._calculate_cache_savings("personalization", personalization_result.get("cost_usd", 0))
                            }
                        )

                        return self._build_response(
                            response=response_text,
                            sources=[],  # From cache, not current RAG
                            cached=True,
                            cache_type="personalized",
                            response_source="cache_personalized",
                            request_start=start_time,
                            model_used=personalization_result.get("model", "personalization"),
                            token_usage={
                                "input_tokens": personalization_result.get("input_tokens", 0),
                                "output_tokens": personalization_result.get("output_tokens", 0),
                                "total_tokens": personalization_result.get("input_tokens", 0) + personalization_result.get("output_tokens", 0)
                            }
                        )

                # Scenario: Cache Hit + Raw (hit_raw)
                response_text = cache_result["response"]
                if isinstance(response_text, list):
                    response_text = str(response_text)
                elif not isinstance(response_text, str):
                    response_text = str(response_text)

                # Use actual sources and token_usage from cache
                cache_sources = cache_result.get("sources", [])
                cache_token_usage = cache_result.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                # Log comprehensive telemetry for raw cache hit
                self.telemetry.log(
                    user_id=request.user_id,
                    method="chat_completion",
                    cache_status="hit_raw",
                    response_source="cache_raw",
                    latency_ms=cache_result.get("latency_ms", 0),
                    input_tokens=cache_token_usage.get("input_tokens", 0),
                    output_tokens=cache_token_usage.get("output_tokens", 0),
                    metadata={
                        "rag_available": bool(rag_chunks),
                        "rag_used": False,
                        "cache_hit": True,
                        "personalization_used": False,
                        "model": "cache",
                        "cache_sources_count": len(cache_sources)
                    }
                )

                return self._build_response(
                    response=response_text,
                    sources=cache_sources,  # ACTUAL sources from cache
                    cached=True,
                    cache_type="raw",
                    response_source="cache_raw",
                    request_start=start_time,
                    model_used="semantic_cache",
                    token_usage=cache_token_usage  # ACTUAL token_usage from cache
                )

            # CACHE MISS scenarios
            if rag_chunks:
                # Scenario: Cache Miss + RAG (knowledge_base)
                try:
                    rag_result = await self.rag_service.generate_answer(
                        question=request.query,
                        context_chunks=rag_chunks,
                        user_memory=request.context
                    )

                    # Extract response and metadata from RAG result
                    rag_response_text = rag_result["response"]
                    rag_cost = rag_result["cost_usd"]
                    rag_latency = rag_result["latency_ms"]
                    rag_input_tokens = rag_result["input_tokens"]
                    rag_output_tokens = rag_result["output_tokens"]
                    rag_model = rag_result["model"]

                    # Store in cache for future with actual sources and token usage
                    await self.semantic_cache.store(
                        query=request.query,
                        response=rag_response_text,
                        user_id=request.user_id,
                        course_id=request.course_id,
                        cache_type="rag_enhanced",
                        sources=rag_chunks,
                        token_usage={
                            "input_tokens": rag_input_tokens,
                            "output_tokens": rag_output_tokens,
                            "total_tokens": rag_input_tokens + rag_output_tokens
                        }
                    )

                    # Log comprehensive telemetry for RAG enhanced response
                    self.telemetry.log(
                        user_id=request.user_id,
                        method="chat_completion",
                        cache_status="miss",
                        response_source="knowledge_base",
                        latency_ms=rag_latency,
                        input_tokens=rag_input_tokens,
                        output_tokens=rag_output_tokens,
                        metadata={
                            "rag_available": True,
                            "rag_used": True,
                            "cache_hit": False,
                            "personalization_used": False,
                            "model": rag_model,
                            "num_sources": len(rag_chunks)
                        }
                    )

                    return self._build_response(
                        response=rag_response_text,
                        sources=[{
                            "id": chunk.id,
                            "content": chunk.content,
                            "distance": chunk.similarity_score,  # RAG uses similarity_score as distance metric
                            "metadata": chunk.metadata
                        } for chunk in rag_chunks],
                        cached=False,
                        cache_type="rag_enhanced",
                        response_source="knowledge_base",
                        request_start=start_time,
                        model_used=rag_model,
                        token_usage={
                            "input_tokens": rag_input_tokens,
                            "output_tokens": rag_output_tokens,
                            "total_tokens": rag_input_tokens + rag_output_tokens
                        }
                    )

                except Exception as e:
                    api_logger.warning("RAG generation failed, falling back to LLM: %s", str(e))
                    # Continue to LLM fallback

            # Scenario: LLM Fallback (llm_fallback)
            llm_response = await self.llm_client.chat_completion(request)

            # Extract actual values from LLM response
            llm_response_text = llm_response.response
            llm_cost = llm_response.metadata.get("cost_usd", 0)
            llm_latency = llm_response.latency_ms
            llm_input_tokens = llm_response.token_usage.get("prompt_tokens", 0)
            llm_output_tokens = llm_response.token_usage.get("completion_tokens", 0)
            llm_model = llm_response.metadata.get("model", "unknown")

            # Store in cache with actual token usage
            await self.semantic_cache.store(
                query=request.query,
                response=llm_response_text,
                user_id=request.user_id,
                course_id=request.course_id,
                cache_type="llm_fallback",
                sources=[],  # No sources for LLM fallback
                token_usage={
                    "input_tokens": llm_input_tokens,
                    "output_tokens": llm_output_tokens,
                    "total_tokens": llm_input_tokens + llm_output_tokens
                }
            )

            # Log comprehensive telemetry for LLM fallback
            self.telemetry.log(
                user_id=request.user_id,
                method="chat_completion",
                cache_status="miss",
                response_source="llm_fallback",
                latency_ms=llm_latency,
                input_tokens=llm_input_tokens,
                output_tokens=llm_output_tokens,
                metadata={
                    "rag_available": False,
                    "rag_used": False,
                    "cache_hit": False,
                    "personalization_used": False,
                    "model": llm_model,
                    "fallback_reason": "no_rag_results"
                }
            )

            return self._build_response(
                response=llm_response_text,
                sources=[],
                cached=False,
                cache_type="llm_fallback",
                response_source="llm_fallback",
                request_start=start_time,
                model_used=llm_model,
                token_usage={
                    "input_tokens": llm_input_tokens,
                    "output_tokens": llm_output_tokens,
                    "total_tokens": llm_input_tokens + llm_output_tokens
                }
            )

        except Exception as e:
            api_logger.error("Chat completion failed: %s", str(e))
            raise TutorServiceException(f"Chat completion failed: {str(e)}")

    async def chat_completion_stream(
        self,
        request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming chat completion with correct RAG + Cache + Personalization flow.

        Flow: RAG preparation → Cache Check → [Hit/Personalize] or [Miss → Generate] → Stream Response
        """
        start_time = time.time()

        if not self._connected:
            await self.connect()

        try:
            # STEP 0: Always try RAG enhancement first (knowledge base retrieval)
            rag_chunks = []
            if not request.skip_rag:
                rag_chunks = await self.rag_service.retrieve_documents(
                    query=request.query,
                    course_id=request.course_id,
                    material_ids=None  # Can be passed from request in future
                )

            # STEP 1: Check semantic cache
            cache_result = await self.semantic_cache.query(
                query=request.query,
                user_id=request.user_id,
                course_id=request.course_id
            )

            if cache_result["cached"] and cache_result["response"]:
                # CACHE HIT scenarios
                if request.context:
                    # Scenario: Cache Hit + Personalization (hit_personalized)
                    personalization_result = await self.semantic_cache.personalize_response(
                        cached_response=cache_result["response"],
                        user_context=request.context,
                        original_query=request.query,
                        user_id=request.user_id
                    )

                    if personalization_result["personalized"]:
                        response_text = personalization_result["response"]
                        if isinstance(response_text, list):
                            response_text = str(response_text)
                        elif not isinstance(response_text, str):
                            response_text = str(response_text)
                        model_used = personalization_result.get("model", "gpt-4o-mini")

                        # Log telemetry for personalized cache hit
                        latency_ms = (time.time() - start_time) * 1000
                        self.telemetry.log(
                            user_id=request.user_id,
                            method="chat_completion_stream",
                            cache_status="hit_personalized",
                            response_source="cache_personalized",
                            latency_ms=latency_ms,
                            input_tokens=personalization_result.get("input_tokens", 0),
                            output_tokens=personalization_result.get("output_tokens", 0),
                            metadata={
                                "rag_available": bool(rag_chunks),
                                "rag_used": False,
                                "cache_hit": True,
                                "personalization_used": True,
                                "model": model_used,
                                "streaming": True
                            }
                        )
                    else:
                        response_text = cache_result["response"]
                        if isinstance(response_text, list):
                            response_text = str(response_text)
                        elif not isinstance(response_text, str):
                            response_text = str(response_text)
                        model_used = "semantic_cache"

                        # Log telemetry for raw cache hit (personalization failed)
                        latency_ms = (time.time() - start_time) * 1000
                        self.telemetry.log(
                            user_id=request.user_id,
                            method="chat_completion_stream",
                            cache_status="hit_raw",
                            response_source="cache_raw",
                            latency_ms=latency_ms,
                            input_tokens=0,
                            output_tokens=0,
                            metadata={
                                "rag_available": bool(rag_chunks),
                                "rag_used": False,
                                "cache_hit": True,
                                "personalization_used": False,
                                "model": "cache",
                                "streaming": True
                            }
                        )
                else:
                    # Scenario: Cache Hit + Raw (hit_raw)
                    response_text = cache_result["response"]
                    if isinstance(response_text, list):
                        response_text = str(response_text)
                    elif not isinstance(response_text, str):
                        response_text = str(response_text)
                    model_used = "semantic_cache"

                    # Log telemetry for raw cache hit
                    latency_ms = (time.time() - start_time) * 1000
                    self.telemetry.log(
                        user_id=request.user_id,
                        method="chat_completion_stream",
                        cache_status="hit_raw",
                        response_source="cache_raw",
                        latency_ms=latency_ms,
                        input_tokens=0,
                        output_tokens=0,
                        metadata={
                            "rag_available": bool(rag_chunks),
                            "rag_used": False,
                            "cache_hit": True,
                            "personalization_used": False,
                            "model": "cache",
                            "streaming": True
                        }
                    )

                # Stream cached response as single chunk with token usage metadata
                yield response_text
                yield f"[TOKEN_USAGE]{{\"model\":\"{model_used}\",\"cached\":true}}[/TOKEN_USAGE]"
                return

            # CACHE MISS scenarios
            if rag_chunks:
                # Scenario: Cache Miss + RAG (knowledge_base)
                try:
                    rag_result = await self.rag_service.generate_answer(
                        question=request.query,
                        context_chunks=rag_chunks,
                        user_memory=request.context
                    )

                    # Extract response and metadata from RAG result
                    rag_response_text = rag_result["response"]
                    rag_cost = rag_result["cost_usd"]
                    rag_latency = rag_result["latency_ms"]
                    rag_input_tokens = rag_result["input_tokens"]
                    rag_output_tokens = rag_result["output_tokens"]
                    rag_model = rag_result["model"]

                    # Store in cache for future
                    await self.semantic_cache.store(
                        query=request.query,
                        response=rag_response_text,
                        user_id=request.user_id,
                        course_id=request.course_id,
                        cache_type="rag_enhanced"
                    )

                    # Log telemetry for RAG enhanced response
                    self.telemetry.log(
                        user_id=request.user_id,
                        method="chat_completion_stream",
                        cache_status="miss",
                        response_source="knowledge_base",
                        latency_ms=rag_latency,
                        input_tokens=rag_input_tokens,
                        output_tokens=rag_output_tokens,
                        metadata={
                            "rag_available": True,
                            "rag_used": True,
                            "cache_hit": False,
                            "personalization_used": False,
                            "model": rag_model,
                            "num_sources": len(rag_chunks),
                            "streaming": True
                        }
                    )

                    # Stream RAG response as single chunk
                    yield rag_response_text
                    yield f"[TOKEN_USAGE]{{\"model\":\"{rag_model}\",\"cached\":false}}[/TOKEN_USAGE]"
                    return

                except Exception as e:
                    api_logger.warning("RAG streaming failed, falling back to LLM: %s", str(e))
                    # Continue to LLM fallback

            # Scenario: LLM Fallback (llm_fallback)
            llm_response = await self.llm_client.chat_completion(request)

            # Extract actual values from LLM response
            llm_response_text = llm_response.response
            llm_cost = llm_response.metadata.get("cost_usd", 0)
            llm_latency = llm_response.latency_ms
            llm_input_tokens = llm_response.token_usage.get("prompt_tokens", 0)
            llm_output_tokens = llm_response.token_usage.get("completion_tokens", 0)
            llm_model = llm_response.metadata.get("model", "unknown")

            # Store in cache
            await self.semantic_cache.store(
                query=request.query,
                response=llm_response_text,
                user_id=request.user_id,
                course_id=request.course_id,
                cache_type="llm_fallback"
            )

            # Log telemetry for LLM fallback
            self.telemetry.log(
                user_id=request.user_id,
                method="chat_completion_stream",
                cache_status="miss",
                response_source="llm_fallback",
                latency_ms=llm_latency,
                input_tokens=llm_input_tokens,
                output_tokens=llm_output_tokens,
                metadata={
                    "rag_available": False,
                    "rag_used": False,
                    "cache_hit": False,
                    "personalization_used": False,
                    "model": llm_model,
                    "fallback_reason": "no_rag_results",
                    "streaming": True
                }
            )

            # Stream LLM response
            yield llm_response_text
            yield f"[TOKEN_USAGE]{{\"model\":\"{llm_model}\",\"cached\":false}}[/TOKEN_USAGE]"

        except Exception as e:
            api_logger.error("Streaming chat completion failed: %s", str(e))
            yield f"Error: {str(e)}"

    def _build_response(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        cached: bool,
        cache_type: str,
        request_start: float,
        model_used: str,
        response_source: str = "unknown",
        token_usage: Optional[Dict[str, int]] = None
    ) -> ChatResponse:
        """Build standardized chat response."""
        latency_ms = (time.time() - request_start) * 1000

        return ChatResponse(
            response=response,
            sources=sources,
            cached=cached,
            token_usage=token_usage or {},  # Use actual token_usage
            latency_ms=round(latency_ms, 2),
            response_source=response_source,
            metadata={
                "cache_type": cache_type,
                "model_used": model_used,
                "optimization_level": "simplified"
            }
        )

    async def get_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        This would need to be implemented based on your storage strategy.
        """
        # TODO: Implement conversation history storage/retrieval
        # This could use Redis key-value storage separate from vector operations
        api_logger.info("Conversation history requested: user_id=%s, limit=%d", user_id, limit)
        return []

    async def clear_conversation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> int:
        """
        Clear conversation history for a user.
        """
        # TODO: Implement conversation history clearing
        api_logger.info("Conversation history clear requested: user_id=%s", user_id)
        return 0

    async def update_user_context(
        self,
        user_id: str,
        context_update: Dict[str, Any]
    ) -> bool:
        """
        Update user context for personalization.
        """
        # TODO: Implement user context storage
        api_logger.info("User context update: user_id=%s, context_keys=%s", user_id, list(context_update.keys()))
        return True

    async def get_chat_analytics(
        self,
        user_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get chat analytics for a user.
        """
        # Get stats from all underlying services
        cache_stats = await self.semantic_cache.get_stats()
        rag_stats = await self.rag_service.get_stats()
        store_stats = await self.vector_store.get_stats()

        return {
            "user_id": user_id,
            "time_window_hours": time_window_hours,
            "semantic_cache": cache_stats.get("semantic_cache", {}),
            "knowledge_base": rag_stats.get("knowledge_base", {}),
            "vector_store_status": store_stats.get("connected", False),
            "service_optimization": "simplified_architecture"
        }

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all service components."""
        try:
            # Check unified vector store (covers both Redis instances)
            vector_store_healthy = await self._check_vector_store_health()

            # Check individual service health
            cache_healthy = bool(self.semantic_cache.llm_client) and bool(self.semantic_cache.vectorizer)
            rag_healthy = bool(self.rag_service.embedding_service) and bool(self.rag_service.llm_client)
            llm_healthy = bool(self.llm_client)

            return {
                "vector_store": vector_store_healthy,
                "semantic_cache": cache_healthy,
                "rag_service": rag_healthy,
                "llm_client": llm_healthy,
                "overall": all([vector_store_healthy, cache_healthy, rag_healthy, llm_healthy])
            }

        except Exception as e:
            api_logger.error("Health check failed: %s", str(e))
            return {
                "vector_store": False,
                "semantic_cache": False,
                "rag_service": False,
                "llm_client": False,
                "overall": False,
                "error": str(e)
            }

    async def _check_vector_store_health(self) -> bool:
        """Check if vector store is healthy."""
        try:
            stats = await self.vector_store.get_stats()
            return stats.get("connected", False)
        except Exception:
            return False

    def _calculate_cache_savings(self, scenario: str, actual_cost: float) -> float:
        """Calculate cost savings compared to baseline LLM call."""
        # Baseline cost for gpt-4o (comprehensive model) ~$0.001 per typical request
        baseline_cost = 0.001

        if scenario == "cache":
            # Full cache hit - 100% savings
            return baseline_cost
        elif scenario == "personalization":
            # Only personalization cost - minimal savings
            return max(0, baseline_cost - actual_cost)
        else:
            # No savings
            return 0