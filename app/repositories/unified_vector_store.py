"""
Unified Knowledge Base Vector Storage - RedisVL Implementation
Handles only knowledge base operations (RAG document storage) using RedisVL best practices.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import redis.asyncio as redis

# RedisVL imports
try:
    from redisvl.index import AsyncSearchIndex
    from redisvl.query import VectorQuery, FilterQuery
    from redisvl.query.filter import Tag, FilterExpression
    from redisvl.schema import IndexSchema
    REDISVL_AVAILABLE = True
except ImportError:
    REDISVL_AVAILABLE = False

from ..config.settings import settings
from ..core.logger import cache_logger
from ..core.exceptions import RedisException


class UnifiedVectorStore:
    """
    Knowledge Base Vector Storage for RAG Document Storage.

    This implementation follows RedisVL best practices:
    - Single global index for all documents
    - Proper filter expressions with Tag and FilterQuery
    - Batch processing with embeddings.aembed_documents
    - Consistent error handling and logging

    Adopted from ai-services vector_store.py with adaptations for tutor-services.
    """

    def __init__(self):
        """Initialize knowledge base vector store with RedisVL configuration."""
        if not REDISVL_AVAILABLE:
            raise RedisException("RedisVL is required for knowledge base operations")

        # Redis configuration for Knowledge Base (port 6379)
        self.knowledge_base_url = settings.redis_knowledge_url
        self.client: redis.Redis = None
        self.index: AsyncSearchIndex = None
        self.index_name = "lms_global_index"
        self.prefix = "chunk"

        # Vector dimensions (will be set from embeddings)
        self.vector_dimension = settings.vector_dimension or 1536
        self.connected = False

        cache_logger.info("UnifiedVectorStore initialized with RedisVL best practices")

    def _define_index_schema(self) -> IndexSchema:
        """Define RedisVL index schema for knowledge base."""
        schema_definition = {
            "index": {
                "name": self.index_name,
                "prefix": self.prefix,
                "storage_type": "hash"
            },
            "fields": [
                # Filterable metadata (Tag)
                {"name": "material_id", "type": "tag"},  # Changed from material_id
                {"name": "course_id", "type": "tag"},
                {"name": "chunk_id", "type": "tag"},  # Added chunk id field

                # Content (Text)
                {"name": "text", "type": "text"},
                {"name": "source_file", "type": "text"},

                # Vector (HNSW)
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "HNSW",
                        "dims": self.vector_dimension,
                        "distance_metric": "COSINE",
                        "datatype": "FLOAT32"
                    }
                }
            ]
        }

        try:
            schema = IndexSchema.from_dict(schema_definition)
            cache_logger.info("RedisVL index schema created successfully")
            return schema
        except Exception as e:
            cache_logger.error(f"Failed to create RedisVL schema: {e}")
            raise RedisException(f"Schema creation failed: {e}")

    async def connect(self) -> None:
        """Connect to Redis and create index if needed."""
        if self.connected and self.index:
            try:
                await self.client.ping()
                return
            except Exception:
                cache_logger.warning("Redis connection lost, reconnecting...")
                self.connected = False
                self.index = None

        try:
            # Connect to Redis
            self.client = await redis.from_url(
                self.knowledge_base_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()

            # Initialize index using external YAML schema (RedisVL best practice)
            try:
                from redisvl.index import AsyncSearchIndex
                self.index = AsyncSearchIndex.from_yaml(
                    "app/schemas/knowledge_base_schema.yaml",
                    redis_client=self.client
                )
                await self.index.create(overwrite=False)
            except Exception as yaml_error:
                cache_logger.warning(f"Failed to load YAML schema: {yaml_error}")
                # Fallback to manual schema definition
                schema = self._define_index_schema()
                self.index = AsyncSearchIndex(schema, redis_client=self.client)
                await self.index.create(overwrite=False)

                self.connected = True
                cache_logger.info(f"Connected to Redis and initialized index '{self.index_name}'")

        except Exception as e:
            if "Index already exists" in str(e):
                cache_logger.info(f"Index '{self.index_name}' already exists")
                # Connect to existing index using YAML schema
                try:
                    self.index = AsyncSearchIndex.from_yaml(
                        "app/schemas/knowledge_base_schema.yaml",
                        redis_client=self.client
                    )
                except Exception as yaml_error:
                    cache_logger.warning(f"Failed to load YAML schema for existing index: {yaml_error}")
                    # Fallback to manual schema
                    schema = self._define_index_schema()
                    self.index = AsyncSearchIndex(schema, redis_client=self.client)
                self.connected = True
            else:
                cache_logger.error(f"Failed to connect to Redis: {e}")
                raise RedisException(f"Connection failed: {e}")

    async def check_document_exists(self, material_id: str) -> bool:
        """
        Check if document with given material_id exists in knowledge base.
        Using RedisVL FilterQuery for proper filtering.
        """
        if not self.connected or not self.index:
            await self.connect()

        try:
            # Use RedisVL FilterQuery for document existence check
            filter_expr = Tag("material_id") == material_id
            query = FilterQuery(
                return_fields=["material_id"],
                filter_expression=filter_expr,
                num_results=1
            )

            result = await self.index.search(query.query, query_params=query.params)
            return len(result.docs) > 0

        except Exception as e:
            cache_logger.error(f"Failed to check document existence: {e}")
            return False

    async def store_documents(
        self,
        texts: List[str],
        vectors: List[List[float]],
        material_id: str,
        course_id: str = "default",
        source_file: str = "unknown"
    ) -> List[str]:
        """
        Store documents in knowledge base for RAG using RedisVL patterns.
        Vectors should already be generated by embedding service.
        """
        if not self.connected or not self.index:
            await self.connect()

        try:
            data_to_load = []

            for i, (text, vector) in enumerate(zip(texts, vectors)):
                # Use RedisVL native format - let RedisVL handle vector conversion
                data = {
                    "text": text,
                    "vector": vector,  # RedisVL handles conversion automatically
                    "material_id": str(material_id),  # Changed from material_id
                    "course_id": str(course_id),
                    "source_file": str(source_file),
                    "chunk_id": f"{material_id}:{i}"  # For chunk identification
                }

                data_to_load.append(data)

            # Store in knowledge base using RedisVL load method
            await self.index.load(data_to_load)

            cache_logger.info(
                "Stored %d document chunks: material_id=%s, course_id=%s",
                len(data_to_load), material_id, course_id
            )
            return [f"{material_id}:{i}" for i in range(len(data_to_load))]

        except Exception as e:
            cache_logger.error("Failed to store documents: %s", str(e))
            raise RedisException(f"Document storage failed: {str(e)}")

    async def search_knowledge_base(
        self,
        query_vector: List[float],
        course_id: Optional[str] = None,
        material_ids: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant documents using RedisVL VectorQuery.
        Follows RedisVL best practices for vector search with course-based filters.
        """
        if not self.connected or not self.index:
            await self.connect()

        try:
            # Build filter expression for course-based and material-based filtering
            filter_expression = None
            if course_id and material_ids:
                # Filter by both course_id AND material_id (most specific)
                from redisvl.query.filter import FilterExpression as FE
                filter_expression = (Tag("course_id") == course_id) & (Tag("material_id") == material_ids[0])
                cache_logger.info("🔍 Using course+material filter: course_id=%s, material_id=%s", course_id, material_ids[0])
            elif course_id:
                # Filter by course_id only (get all materials in course)
                filter_expression = Tag("course_id") == course_id
                cache_logger.info("🔍 Using course filter: course_id=%s", course_id)
            elif material_ids:
                # Fallback to material_id filtering only
                filter_expression = Tag("material_id") == material_ids[0]
                cache_logger.info("🔍 Using material filter: material_id=%s", material_ids[0])
            else:
                cache_logger.info("🔍 No filters - searching all documents")

            # Create VectorQuery following RedisVL best practices
            vector_query = VectorQuery(
                vector=query_vector,
                vector_field_name="vector",
                return_fields=["text", "material_id", "course_id", "source_file", "chunk_id"],
                num_results=top_k,
                return_score=True
            )

            # Apply filter expression if we have one (course-based or file-based)
            if filter_expression:
                vector_query.filter_expression = filter_expression
                cache_logger.info("🔍 Applied filter expression to vector query")
            elif material_ids:
                # Fallback to legacy file hash filtering
                vector_query.filter_expression = self._build_filter_expression(material_ids)
                cache_logger.info("🔍 Applied legacy material filter expression")

            # Execute search
            result = await self.index.search(vector_query.query, query_params=vector_query.params)

            cache_logger.info(
                "🔍 VECTOR SEARCH RESULTS: found %d documents total",
                len(result.docs)
            )

            # Process results with COSINE DISTANCE filtering
            documents = []
            for doc in result.docs:
                vector_score = getattr(doc, "vector_distance", 1.0)
                # Convert to float if it's a string
                try:
                    distance = float(vector_score) if vector_score is not None else 1.0
                except (ValueError, TypeError):
                    distance = 1.0

                cache_logger.info(
                    "   📄 Document: material_id=%s, vector_distance=%.4f, threshold=%.2f, passes=%s",
                    getattr(doc, "material_id", "unknown"),
                    distance,
                    threshold,
                    distance <= threshold
                )

                # RedisVL VectorQuery returns COSINE DISTANCE
                # Lower distance = more similar (0.0 = perfect match, 1.0 = completely different)
                if distance <= threshold:
                    documents.append({
                        "text": getattr(doc, "text", ""),
                        "material_id": getattr(doc, "material_id", ""),
                        "course_id": getattr(doc, "course_id", ""),
                        "source_file": getattr(doc, "source_file", ""),
                        "chunk_id": getattr(doc, "chunk_id", ""),
                        "vector_distance": distance
                    })

            return documents

        except Exception as e:
            cache_logger.error("Failed to search knowledge base: %s", str(e))
            return []

    def _build_filter_expression(self, material_ids: List[str]) -> Optional[FilterExpression]:
        """
        Build filter OR expression for multiple file hashes using RedisVL Filter API.
        (Tag("material_id") == "materi_A") | (Tag("material_id") == "materi_B")
        """
        if not material_ids:
            return None

        # Create individual filters and combine with OR
        filters = [Tag("material_id") == h for h in material_ids]

        # Combine filters with OR operator
        filter_expr = filters[0]
        for f in filters[1:]:
            filter_expr = filter_expr | f

        return filter_expr

    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.connected:
            await self.connect()

        try:
            # Get total entries count
            knowledge_keys = await self.client.keys(f"{self.prefix}:*")

            return {
                "knowledge_base": {
                    "type": "document_store",
                    "redis_url": self.knowledge_base_url,
                    "index_name": self.index_name,
                    "prefix": self.prefix,
                    "entries": len(knowledge_keys),
                    "purpose": "RAG knowledge retrieval"
                },
                "vector_dimension": self.vector_dimension,
                "redisvl_available": REDISVL_AVAILABLE,
                "connected": self.connected
            }

        except Exception as e:
            cache_logger.error("Failed to get stats: %s", str(e))
            return {"error": str(e)}

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        try:
            if self.client:
                await self.client.close()
            self.connected = False
            cache_logger.info("UnifiedVectorStore disconnected from Redis")

        except Exception as e:
            cache_logger.error("Error during disconnect: %s", str(e))


# Global singleton instance
unified_vector_store = UnifiedVectorStore()