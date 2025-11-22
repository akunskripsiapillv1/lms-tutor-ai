"""
Simple Semantic Cache Service using RedisVL SemanticCache
Semantic cache using Redis (port 6380) with proper acheck/astore methods
"""
import time
import os
from typing import Dict, Any, Optional
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import OpenAITextVectorizer
from ..config.settings import settings
from ..core.logger import cache_logger

class SimpleCacheService:
    """
    Simple semantic cache service using RedisVL SemanticCache
    Uses acheck() and astore() methods for proper semantic caching
    """

    def __init__(self):
        """Initialize the cache service with RedisVL SemanticCache"""
        # Ensure OpenAI API key is available
        if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        # Initialize OpenAI vectorizer
        self.vectorizer = OpenAITextVectorizer(
            model=settings.openai_embedding_model
        )

        # RedisVL Semantic Cache (port 6380)
        self.semantic_cache = SemanticCache(
            name="semantic_cache",
            redis_url=settings.redis_cache_url,  # Semantic cache port from settings
            vectorizer=self.vectorizer,
            distance_threshold=0.05,  # Lower threshold = less sensitive (smaller distance = more similar)
            ttl=settings.cache_ttl,  # TTL from settings
            overwrite=True
        )

        cache_logger.info("SimpleCacheService initialized with RedisVL SemanticCache (port 6380)")

    async def get(self, query: str, user_id: Optional[str] = None, course_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached response using acheck() method

        Args:
            query: The query to check in cache
            user_id: Optional user ID for context
            course_id: Optional course ID for context

        Returns:
            Cached response or None if not found
        """
        try:
            start_time = time.time()

            # Create context-aware cache key
            context_suffix = ""
            if user_id:
                context_suffix += f"|user:{user_id}"
            if course_id:
                context_suffix += f"|course:{course_id}"

            enhanced_query = f"{query}{context_suffix}" if context_suffix else query

            cache_logger.info(f"🔍 Checking cache: original_query='{query[:30]}...', enhanced_query='{enhanced_query[:30]}...', user_id={user_id}, course_id={course_id}")

            # Generate embedding vector for the query for better control
            query_vector = await self.vectorizer.aembed(enhanced_query)

            cache_logger.info(f"🔍 Generated query vector (dimension: {len(query_vector)})")

            # Use acheck() with vector parameter for more control
            result = await self.semantic_cache.acheck(
                vector=query_vector,
                distance_threshold=0.01,  # Much stricter threshold
                num_results=1,
                return_fields=["response", "metadata", "prompt"]  # Add 'prompt' field to see cached query
            )

            latency = (time.time() - start_time) * 1000

            cache_logger.info(f"🔍 Cache check result: found={len(result) if result else 0} entries")

            if result and len(result) > 0:
                cache_entry = result[0]
                response_text = cache_entry.get("response", "")
                metadata = cache_entry.get("metadata", {})

                # Get distance/score if available
                distance = getattr(cache_entry, 'vector_distance', 'N/A')
                if hasattr(cache_entry, 'vector_distance'):
                    distance = f"{cache_entry.vector_distance:.4f}"

                cache_logger.info(f"🎯 Cache HIT COMPARISON:")
                cache_logger.info(f"   User Query: '{query}'")
                cache_logger.info(f"   Enhanced Query: '{enhanced_query}'")

                # Get the cached prompt (original query that was cached)
                cached_prompt = cache_entry.get("prompt", "N/A")
                cache_logger.info(f"   Cached Query (from prompt field): '{cached_prompt}'")

                cache_logger.info(f"   Cached Response (first 100 chars): '{response_text[:100]}...'")

                # Debug: Print all available attributes and keys
                cache_logger.info(f"   Cache Entry Type: {type(cache_entry)}")
                cache_logger.info(f"   Available Attributes: {[attr for attr in dir(cache_entry) if not attr.startswith('_')]}")

                if hasattr(cache_entry, '__dict__'):
                    cache_logger.info(f"   Entry Dict Keys: {list(cache_entry.__dict__.keys())}")

                if isinstance(cache_entry, dict):
                    cache_logger.info(f"   Dict Keys: {list(cache_entry.keys())}")
                    for key, value in cache_entry.items():
                        if key not in ['response', 'metadata', 'prompt']:
                            cache_logger.info(f"     {key}: {value}")

                cache_logger.info(f"   Distance: {distance}")
                cache_logger.info(f"   Cached Metadata: {metadata}")

                cache_logger.info(f"Cache HIT: query_length={len(query)}, latency_ms={latency:.2f}")

                return {
                    "response": response_text,
                    "cached": True,
                    "cache_type": "hit",
                    "user_id": user_id or "anonymous",
                    "course_id": course_id,
                    "latency_ms": latency,
                    "sources": metadata.get("sources", [])
                }
            else:
                cache_logger.debug(f"Cache MISS: query_length={len(query)}")
                return None

        except Exception as e:
            cache_logger.error(f"Cache lookup failed: {e}")
            return None

    async def set(self, query: str, response: str, user_id: Optional[str] = None,
                  course_id: Optional[str] = None, sources: Optional[list] = None) -> bool:
        """
        Store response using astore() method

        Args:
            query: The original query
            response: The response to cache
            user_id: Optional user ID
            course_id: Optional course ID
            sources: Optional source documents

        Returns:
            True if successfully cached
        """
        try:
            # Create context-aware cache key (same as get method)
            context_suffix = ""
            if user_id:
                context_suffix += f"|user:{user_id}"
            if course_id:
                context_suffix += f"|course:{course_id}"

            enhanced_query = f"{query}{context_suffix}" if context_suffix else query

            metadata = {
                "user_id": user_id or "anonymous",
                "course_id": course_id or "global",
                "sources": sources or [],
                "timestamp": time.time()
            }

            # Use astore() to cache the response with enhanced query
            await self.semantic_cache.astore(
                prompt=enhanced_query,
                response=response,
                metadata=metadata
            )

            cache_logger.debug(f"Cached response: query_length={len(query)}, response_length={len(response)}")
            return True

        except Exception as e:
            cache_logger.error(f"Cache storage failed: {e}")
            return False

    async def clear_user_cache(self, user_id: str) -> int:
        """Clear all cached entries for a specific user"""
        try:
            # TODO: Implement user-based cache clearing using RedisVL methods
            cache_logger.info(f"Clearing cache for user: {user_id}")
            return 0
        except Exception as e:
            cache_logger.error(f"Failed to clear user cache: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return {
                "cache_name": "semantic_cache",
                "redis_url": "redis://localhost:6380",
                "distance_threshold": 0.1,
                "ttl": settings.cache_ttl,
                "embedding_model": settings.openai_embedding_model,
                "status": "active",
                "note": "SemanticCache initialized successfully"
            }
        except Exception as e:
            cache_logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}