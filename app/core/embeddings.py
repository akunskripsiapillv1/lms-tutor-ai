from typing import List, Dict, Any, Optional
import numpy as np

from ..config.settings import settings
from .exceptions import EmbeddingException
from .logger import model_logger

# Updated imports based on latest RedisVL patterns
try:
    from redisvl.utils.vectorize import OpenAITextVectorizer
    REDISVL_AVAILABLE = True
except ImportError:
    REDISVL_AVAILABLE = False
    model_logger.warning("RedisVL not fully available, using fallback implementations")


class EmbeddingService:
    """
    RedisVL Native Embedding Service using OpenAITextVectorizer.

    This implementation follows RedisVL best practices:
    - Native OpenAITextVectorizer with async support (aembed, aembed_many)
    - Proper API configuration and error handling
    - Optimized batch processing
    - No manual vector calculations

    Replaces: Custom embedding implementations with RedisVL native methods
    """

    def __init__(self):
        """
        Initialize embedding service with RedisVL OpenAITextVectorizer.

        Following RedisVL documentation patterns for OpenAI embedding configuration.
        """
        # Validate required configuration
        if not bool(settings.openai_api_key):
            raise EmbeddingException("OpenAI API key is required for embedding service")

        if not REDISVL_AVAILABLE:
            raise EmbeddingException("RedisVL is required for embedding service")

        # Initialize OpenAI vectorizer following RedisVL best practices
        try:
            # Configure API settings (without model to avoid conflict)
            api_config = {
                "api_key": settings.openai_api_key
            }

            # Initialize OpenAI vectorizer
            self.vectorizer = OpenAITextVectorizer(
                model=settings.openai_embedding_model,
                api_config=api_config
            )

            # Get embedding dimension from vectorizer
            self.embedding_dimension = self.vectorizer.dims

            model_logger.info(
                "OpenAI vectorizer initialized successfully: model=%s, dimensions=%s",
                settings.openai_embedding_model,
                self.embedding_dimension
            )

        except Exception as e:
            model_logger.error(
                "Failed to initialize OpenAI vectorizer: model=%s, error=%s",
                settings.openai_embedding_model, str(e)
            )
            raise EmbeddingException(f"Failed to initialize OpenAI embedding model: {str(e)}")

        model_logger.info(
            "EmbeddingService initialized with RedisVL OpenAI vectorizer: model=%s, dimensions=%s",
            settings.openai_embedding_model,
            self.embedding_dimension
        )

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using RedisVL OpenAITextVectorizer.

        Following RedisVL async patterns with proper error handling.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingException: If text is empty or embedding generation fails
        """
        # Input validation
        if not text or not text.strip():
            raise EmbeddingException("Text cannot be empty")

        try:
            # Use RedisVL's native async method
            if hasattr(self.vectorizer, 'aembed'):
                # Preferred: Use async method for better performance
                embedding = await self.vectorizer.aembed(text)
                model_logger.debug(
                    "Generated embedding using aembed: text_length=%d, dimensions=%d",
                    len(text), len(embedding)
                )
            else:
                # Fallback: Use sync method (older RedisVL versions)
                embedding = self.vectorizer.embed(text)
                model_logger.debug(
                    "Generated embedding using embed: text_length=%d, dimensions=%d",
                    len(text), len(embedding)
                )

            return embedding

        except Exception as e:
            model_logger.error(
                "OpenAI embedding generation failed: text_length=%d, error=%s",
                len(text), str(e)
            )
            raise EmbeddingException(f"Failed to generate OpenAI embedding: {str(e)}")

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using RedisVL batch processing.

        Following RedisVL documentation for optimal batch processing with aembed_many.

        Args:
            texts: List of input texts to embed
            batch_size: Batch size for processing (default: 100, max: 2048 for OpenAI)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingException: If batch embedding generation fails
        """
        # Input validation
        if not texts:
            return []

        if not isinstance(texts, list):
            raise EmbeddingException("Input must be a list of strings")

        # Validate batch size according to OpenAI limits
        if batch_size > 2048:
            batch_size = 2048
            model_logger.warning("Batch size reduced to 2048 (OpenAI limit)")

        try:
            # Use RedisVL's native async batch method
            if hasattr(self.vectorizer, 'aembed_many'):
                # Preferred: Use async batch method for optimal performance
                embeddings = await self.vectorizer.aembed_many(texts, batch_size=batch_size)
                model_logger.debug(
                    "Generated batch embeddings using aembed_many: texts=%d, batch_size=%d, output_dimensions=%d",
                    len(texts), batch_size, len(embeddings) if embeddings else 0
                )
            else:
                # Fallback: Use sync batch method
                embeddings = self.vectorizer.embed_many(texts, batch_size=batch_size)
                model_logger.debug(
                    "Generated batch embeddings using embed_many: texts=%d, batch_size=%d, output_dimensions=%d",
                    len(texts), batch_size, len(embeddings) if embeddings else 0
                )

            return embeddings

        except Exception as e:
            model_logger.error(
                "Batch embedding generation failed: texts_count=%d, batch_size=%d, error=%s",
                len(texts), batch_size, str(e)
            )
            raise EmbeddingException(f"Failed to generate batch embeddings: {str(e)}")

    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate embedding vector format and dimensions following RedisVL patterns.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if valid
        """
        try:
            # Check type
            if not isinstance(embedding, list):
                model_logger.debug("Embedding validation failed: not a list, type=%s", type(embedding))
                return False

            # Check dimensions
            if len(embedding) != self.embedding_dimension:
                model_logger.debug(
                    "Embedding validation failed: dimensions mismatch, expected=%d, actual=%d",
                    self.embedding_dimension, len(embedding)
                )
                return False

            # Check for NaN or infinite values
            embedding_array = np.array(embedding)
            if not np.all(np.isfinite(embedding_array)):
                model_logger.debug("Embedding validation failed: contains NaN or infinite values")
                return False

            return True

        except Exception as e:
            model_logger.debug("Embedding validation failed with exception: %s", str(e))
            return False

    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings from RedisVL vectorizer.

        Returns:
            Embedding dimension
        """
        return self.embedding_dimension

    async def health_check(self) -> bool:
        """
        Check if embedding service is healthy using RedisVL vectorizer.

        Returns:
            True if service is healthy and can generate embeddings
        """
        try:
            # Test embedding generation with simple text
            test_text = "Hello, world!"
            embedding = await self.generate_embedding(test_text)

            # Validate the generated embedding
            is_valid = self.validate_embedding(embedding)

            if is_valid:
                model_logger.debug("Embedding service health check passed")
            else:
                model_logger.warning("Embedding service health check failed: invalid embedding generated")

            return is_valid

        except Exception as e:
            model_logger.error("Embedding service health check failed: %s", str(e))
            return False

    def get_vectorizer_info(self) -> Dict[str, Any]:
        """
        Get detailed information about RedisVL vectorizer configuration.

        Returns:
            Dictionary with vectorizer configuration and capabilities
        """
        try:
            info = {
                "embedding_model": settings.openai_embedding_model,
                "embedding_dimension": self.embedding_dimension,
                "redisvl_available": REDISVL_AVAILABLE,
                "async_support": hasattr(self.vectorizer, 'aembed'),
                "batch_support": hasattr(self.vectorizer, 'aembed_many'),
                "vectorizer_type": "OpenAITextVectorizer"
            }

            # Add additional info if available
            if hasattr(self.vectorizer, 'model'):
                info["model_details"] = str(self.vectorizer.model)

            return info

        except Exception as e:
            model_logger.error("Failed to get vectorizer info: %s", str(e))
            return {"error": str(e), "redisvl_available": REDISVL_AVAILABLE}