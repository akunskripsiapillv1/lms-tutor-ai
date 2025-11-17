"""
Simplified Semantic Cache - Native RedisVL Implementation

This replaces the manual vector operations with RedisVL's native SemanticCache
for better performance and simplicity.
"""
from typing import Dict, Any, Optional, List
import time
import os

from ..config.settings import settings
from ..core.llm_client import LLMClient
from ..core.telemetry import TelemetryLogger, TokenCounter
from ..core.exceptions import CacheException
from ..core.logger import cache_logger
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import OpenAITextVectorizer


class SimplifiedSemanticCache:
    """
    Simplified semantic cache using RedisVL's native SemanticCache with OpenAI embeddings.

    This replaces all manual vector operations with RedisVL's built-in functionality:
    - Automatic embedding generation with OpenAI
    - Native similarity search with distance_threshold
    - Built-in TTL and metadata management
    - No manual vector calculations needed

    Replaces: ContextEnabledSemanticCache (complex manual implementation)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        telemetry: Optional[TelemetryLogger] = None
    ):
        """Initialize simplified semantic cache with native RedisVL."""
        self.llm_client = llm_client or LLMClient()
        self.telemetry = telemetry or TelemetryLogger()

        # Configuration
        # RedisVL SemanticCache uses distance_threshold (lower = more similar)
        # Use threshold directly as distance_threshold (0.0-1.0, lower = more similar)
        self.distance_threshold = settings.cache_threshold
        self.cache_name = "semantic_cache"
        self.redis_url = settings.redis_cache_url  # Semantic cache on port 6380
        self.cache_ttl = settings.cache_ttl

        # Ensure OpenAI API key is available in environment for RedisVL
        if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        # Initialize OpenAI vectorizer
        self.vectorizer = OpenAITextVectorizer(
            model=settings.openai_embedding_model
        )

        # Initialize token counter for accurate token counting
        self.token_counter = TokenCounter()

        # Initialize RedisVL SemanticCache with proper distance_threshold
        # RedisVL uses distance_threshold (lower = more similar)
        self.semantic_cache = SemanticCache(
            name=self.cache_name,
            redis_url=self.redis_url,
            distance_threshold=self.distance_threshold,
            vectorizer=self.vectorizer,
            ttl=self.cache_ttl,
            overwrite=True
        )

        cache_logger.info(
            "SimplifiedSemanticCache initialized with distance-based logic: distance_threshold=%.2f, model=%s, ttl=%s",
            self.distance_threshold, settings.openai_embedding_model, self.cache_ttl
        )

    def _should_personalize(self, user_id: Optional[str], course_id: Optional[str],
                           cache_entry: Dict[str, Any]) -> bool:
        """
        Determine if cached response needs personalization.

        Returns True if:
        - User has specific context (goals, preferences, history)
        - Course-specific personalization is needed
        - Cache entry lacks user-specific information
        """
        if not user_id or user_id == "anonymous":
            return False

        # Check if cache entry already has user-specific metadata
        cache_metadata = cache_entry.get("metadata", {})
        cached_user_id = cache_metadata.get("user_id")
        cached_course_id = cache_metadata.get("course_id")

        # If cached for different user or different course, need personalization
        if cached_user_id != user_id or cached_course_id != course_id:
            return True

        # If cache lacks user context, might need personalization
        if not cache_metadata.get("has_user_context", False):
            return True

        return False

    async def query(self, query: str, user_id: Optional[str] = None, course_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query semantic cache for similar responses using native RedisVL SemanticCache.
        Returns unified response format with course-based isolation.
        """
        start_time = time.time()

        try:
            # Use RedisVL SemanticCache native async check method
            # RedisVL handles embedding generation and similarity search automatically
            # Note: SemanticCache uses its own prefix, not chunk prefix like VectorStore
            # Use basic acheck call without custom fields for simplicity
            result = await self.semantic_cache.acheck(
                prompt=query
            )

            latency = (time.time() - start_time) * 1000

            # Comprehensive debug logging
            cache_logger.info(
                "🔍 CACHE DEBUG: Query='%s...', latency_ms=%.2f",
                query[:100] + "..." if len(query) > 100 else query,
                latency
            )

            cache_logger.debug(
                "RedisVL acheck result: type=%s, result=%s, length=%s",
                type(result).__name__,
                str(result)[:500] if result else "None",  # Longer for debugging
                len(result) if isinstance(result, (list, dict)) else "N/A"
            )

            # Check if we got a cache hit - RedisVL SemanticCache always returns list of cache entries
            if result and isinstance(result, list) and len(result) > 0:
                # Get the first (most similar) cache entry
                cache_entry = result[0]

                # Debug: Log complete cache entry structure
                cache_logger.debug(
                    "Cache entry analysis: keys=%s, entry_preview=%s",
                    list(cache_entry.keys()) if cache_entry else [],
                    str(cache_entry)[:800] if cache_entry else "None"
                )

                # Get vector distance from RedisVL SemanticCache
                vector_distance = float(cache_entry.get("vector_distance", 0.0))

                # Debug: Log calculation
                cache_logger.debug(
                    "Cache vector_distance=%.6f, distance_threshold=%.6f",
                    vector_distance, self.distance_threshold
                )

                # RedisVL SemanticCache: vector_distance
                # Lower vector_distance = more similar
                meets_threshold = vector_distance <= self.distance_threshold
                cache_logger.debug(
                    "Check: vector_distance=%.6f <= threshold=%.6f = %s",
                    vector_distance, self.distance_threshold, meets_threshold
                )

                if meets_threshold:
                    # Extract response from cache entry
                    response_text = cache_entry.get("response", "")

                    cache_logger.info(
                        "✅ CACHE HIT: vector_distance=%.4f <= threshold=%.4f, response_length=%d",
                        vector_distance, self.distance_threshold, len(response_text)
                    )

                    # Debug: Log _should_personalize parameters and result
                    cache_logger.debug(
                        "Personalization check: user_id=%s, course_id=%s, cache_keys=%s",
                        user_id, course_id, list(cache_entry.keys()) if cache_entry else []
                    )

                    needs_personalization = self._should_personalize(user_id, course_id, cache_entry)
                    cache_type = "hit_personalized" if needs_personalization else "hit_raw"

                    cache_logger.debug(
                        "Personalization result: needs_personalize=%s, cache_type=%s",
                        needs_personalization, cache_type
                    )

                    # Extract stored metadata from cache entry
                    cache_metadata = cache_entry.get("metadata", {})

                    cache_logger.debug(
                        "Cache metadata: metadata_keys=%s, sources_count=%d, token_usage=%s",
                        list(cache_metadata.keys()) if cache_metadata else [],
                        len(cache_metadata.get("sources", [])),
                        cache_metadata.get("token_usage", {})
                    )

                    cache_result = {
                        "response": response_text,
                        "cached": True,
                        "cache_type": cache_type,
                        "vector_distance": vector_distance,
                        "source": "semantic_cache",
                        "user_id": user_id or "anonymous",
                        "sources": cache_metadata.get("sources", []),
                        "token_usage": cache_metadata.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                    }
                else:
                    cache_logger.warning(
                        "❌ CACHE MISS: vector_distance=%.4f > threshold=%.4f",
                        vector_distance, self.distance_threshold
                    )

                    cache_result = {
                        "response": None,
                        "cached": False,
                        "cache_type": "miss",
                        "vector_distance": vector_distance,  # Keep original name
                        "source": "semantic_cache"
                    }
            else:
                # Cache miss - no similar entries found
                cache_result = {
                    "response": None,
                    "cached": False,
                    "cache_type": "miss",
                    "vector_distance": 1.0,
                    "source": "semantic_cache"
                }

            # Count actual tokens using TokenCounter
            query_tokens = self.token_counter.count_tokens(query)
            response_tokens = self.token_counter.count_tokens(cache_result.get("response", "")) if cache_result["cached"] else 0

            # Log telemetry
            self.telemetry.log(
                user_id=user_id or "anonymous",
                method="semantic_cache_query",
                latency_ms=latency,
                input_tokens=query_tokens,
                output_tokens=response_tokens,
                cache_status="hit" if cache_result["cached"] else "miss",
                response_source="semantic_cache",
                metadata={
                    "cache_type": cache_result["cache_type"],
                    "vector_distance": cache_result.get("vector_distance", 1),
                    "query_length": len(query)
                }
            )

            # Reduce verbosity - only log cache hits or issues
            if cache_result["cached"]:
                cache_logger.info(
                    "Cache HIT: query_length=%d, vector_distance=%.6f, latency_ms=%.2f",
                    len(query),
                    cache_result.get("vector_distance", 0),
                    latency
                )
            else:
                cache_logger.debug(
                    "Cache MISS: query_length=%d, latency_ms=%.2f",
                    len(query),
                    latency
                )

            return cache_result

        except Exception as e:
            cache_logger.error("Semantic cache query failed: %s", str(e))
            raise CacheException(f"Cache query failed: {str(e)}")

    async def store(
        self,
        query: str,
        response: str,
        user_id: Optional[str] = None,
        course_id: Optional[str] = None,
        cache_type: str = "response",
        sources: Optional[List[Dict[str, Any]]] = None,
        token_usage: Optional[Dict[str, int]] = None
    ) -> bool:
        """
        Store query-response pair in semantic cache using native RedisVL SemanticCache.

        Args:
            query: The original query
            response: The response text
            user_id: User identifier
            course_id: Course identifier
            cache_type: Type of cache entry (rag_enhanced, llm_fallback, etc.)
            sources: RAG document sources if applicable
            token_usage: Token usage statistics
        """
        start_time = time.time()

        try:
            # Prepare essential metadata with actual values
            metadata = {
                "user_id": user_id or "anonymous",
                "course_id": course_id or "global",
                "cache_type": cache_type,
                "created_at": int(time.time()),
                "sources": sources or [],
                "token_usage": token_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            }

            # Use RedisVL SemanticCache native async store method
            await self.semantic_cache.astore(
                prompt=query,
                response=response,
                metadata=metadata
            )

            latency = (time.time() - start_time) * 1000

            # Count actual tokens using TokenCounter
            query_tokens = self.token_counter.count_tokens(query)
            response_tokens = self.token_counter.count_tokens(response)

            # Log telemetry
            self.telemetry.log(
                user_id=user_id or "anonymous",
                method="semantic_cache_store",
                latency_ms=latency,
                input_tokens=query_tokens,
                output_tokens=response_tokens,
                cache_status="na",  # Storage operations are "not applicable" for cache status
                response_source="semantic_cache",
                metadata={
                    "cache_type": cache_type,
                    "query_length": len(query),
                    "response_length": len(response)
                }
            )

            # Store operations are routine, log at debug level
            cache_logger.debug(
                "Cache store: query_length=%d, response_length=%d, cache_type=%s, latency_ms=%.2f",
                len(query),
                len(response),
                cache_type,
                latency
            )

            return True

        except Exception as e:
            cache_logger.error("Semantic cache store failed: %s", str(e))
            raise CacheException(f"Cache store failed: {str(e)}")

    async def personalize_response(
        self,
        cached_response: str,
        user_context: Dict[str, Any],
        original_query: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Personalize cached response using LLM client.
        Business logic only - storage handled by vector store.
        """
        start_time = time.time()

        try:
            # Use LLM client for personalization
            result = await self.llm_client.personalize_response(
                cached_response=cached_response,
                user_context=user_context,
                original_prompt=original_query
            )

            latency = (time.time() - start_time) * 1000

            # Store personalized response in cache
            if result.get("response"):
                # Ensure response is string before storing
                response_text = result["response"]
                if isinstance(response_text, list):
                    response_text = str(response_text)
                elif not isinstance(response_text, str):
                    response_text = str(response_text)

                # Use RedisVL SemanticCache astore method directly
                await self.semantic_cache.astore(
                    prompt=original_query,
                    response=response_text,
                    metadata={
                        "user_id": user_id or "anonymous",
                        "cache_type": "personalized",
                        "created_at": int(time.time())
                    }
                )

            # Log telemetry
            self.telemetry.log(
                user_id=user_id or "anonymous",
                method="response_personalization",
                latency_ms=latency,
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                cache_status="personalized",  # Response personalization from cached content
                response_source=result.get("model", "unknown"),
                metadata={
                    "original_response_length": len(cached_response),
                    "personalized_response_length": len(result.get("response", "")),
                    "has_user_context": bool(user_context)
                }
            )

            cache_logger.info(
                "Response personalized: input_tokens=%d, output_tokens=%d, latency_ms=%.2f",
                result.get("input_tokens", 0),
                result.get("output_tokens", 0),
                latency
            )

            return {
                "response": result["response"],
                "personalized": True,
                "cache_type": "personalized",
                "latency_ms": result.get("latency_ms", latency),
                "model": result.get("model", "unknown")
            }

        except Exception as e:
            cache_logger.error("Response personalization failed: %s", str(e))
            # Return original response if personalization fails
            return {
                "response": cached_response,
                "personalized": False,
                "cache_type": "fallback",
                "error": str(e)
            }

    async def clear_user_cache(self, user_id: str) -> int:
        """Clear all cache entries for specific user using native RedisVL."""
        try:
            # Get all keys for this user using filter query
            from redisvl.query.filter import Tag
            from redisvl.query import FilterQuery

            filter_expr = Tag("user_id") == user_id
            query = FilterQuery(
                return_fields=["user_id"],
                filter_expression=filter_expr,
                num_results=1000  # Adjust based on expected cache size
            )

            result = await self.semantic_cache.index.search(query.query, query_params=query.params)

            # Delete all matching keys
            keys_deleted = 0
            for doc in result.docs:
                key = doc.get("id", "")
                if key:
                    await self.semantic_cache.index.client.delete(key)
                    keys_deleted += 1

            cache_logger.info("Cleared user cache: user_id=%s, keys_deleted=%d", user_id, keys_deleted)
            return keys_deleted

        except Exception as e:
            cache_logger.error("Failed to clear user cache: user_id=%s, error=%s", user_id, str(e))
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics from native RedisVL SemanticCache."""
        try:
            # Get stats from RedisVL SemanticCache directly
            cache_info = self.semantic_cache.index.info()

            # Get total number of entries using proper RedisVL method
            total_entries = len(await self.semantic_cache.index.search("*"))

            return {
                "semantic_cache": {
                    "name": self.cache_name,
                    "redis_url": self.redis_url,
                    "entries": total_entries,
                    "index_name": cache_info.get("index_name", "unknown"),
                    "vector_dimension": cache_info.get("index_info", {}).get("dimensions", "unknown")
                },
                "distance_threshold": self.distance_threshold,
                "cache_ttl": self.cache_ttl,
                "embedding_model": settings.openai_embedding_model,
                "llm_client_initialized": bool(self.llm_client),
                "distance_based_logic": True,
                "redisvl_version": "latest"
            }

        except Exception as e:
            cache_logger.error("Failed to get semantic cache stats: %s", str(e))
            return {"error": str(e)}