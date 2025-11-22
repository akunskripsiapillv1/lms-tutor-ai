from typing import Dict, Any, List, Optional, AsyncGenerator
from openai import AsyncOpenAI, OpenAI
import time

from ..config.settings import settings
from .exceptions import LLMException
from .telemetry import TokenCounter
from ..models.chat import ChatRequest, ChatResponse
from .logger import model_logger


class LLMClient:
    """Client for interacting with OpenAI LLM with flexible model strategy."""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """
        Initialize LLM client with OpenAI connection.

        Args:
            token_counter: Optional token counter for usage tracking
        """
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.token_counter = token_counter or TokenCounter()

        # Flexible model configuration for different use cases
        self.models = {
            "comprehensive": settings.openai_model or "gpt-4o",  # Default comprehensive model
            "personalization": "gpt-4o-mini",  # For response personalization
            "efficient": "gpt-4o-mini",  # General efficient tasks
            "fallback": "gpt-3.5-turbo",  # Cost-effective fallback
            "lightweight": "gpt-4o-mini"  # Ultra-lightweight tasks
        }

        # Support for custom model overrides via settings
        self.models.update({
            "custom_comprehensive": getattr(settings, 'custom_comprehensive_model', None),
            "custom_personalization": getattr(settings, 'custom_personalization_model', None),
            "custom_fallback": getattr(settings, 'custom_fallback_model', None)
        })

        # Filter out None values
        self.models = {k: v for k, v in self.models.items() if v is not None}

        model_logger.info("LLMClient initialized with flexible models: %s", self.models)

    def call_llm(self, prompt: str, model: Optional[str] = None, max_tokens: int = 200, use_case: Optional[str] = None) -> Dict[str, Any]:
        """
        Call LLM model with flexible model selection.

        Args:
            prompt: Input prompt for the model
            model: Specific model to use (overrides use_case)
            max_tokens: Maximum tokens in response
            use_case: Use case type ("comprehensive", "personalization", "efficient", "fallback", "lightweight")

        Returns:
            Dictionary with response and metadata
        """
        # Model selection priority: explicit model > use_case > default comprehensive
        if model:
            selected_model = model
        elif use_case and use_case in self.models:
            selected_model = self.models[use_case]
        else:
            selected_model = self.models["comprehensive"]
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )

            latency = (time.time() - start_time) * 1000  # Convert to milliseconds

            output = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Use TokenCounter for accurate cost calculation
            cost = self.token_counter.calculate_cost(input_tokens, output_tokens, selected_model)

            model_logger.info(
                "LLM call completed: model=%s, use_case=%s, latency_ms=%s, input_tokens=%s, output_tokens=%s, cost_usd=%s",
                selected_model, use_case, round(latency, 2), input_tokens, output_tokens, cost
            )

            return {
                "response": output,
                "latency_ms": round(latency, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model": selected_model,
                "use_case": use_case,
                "cost_usd": cost
            }

        except Exception as e:
            model_logger.error("LLM call failed: model=%s, use_case=%s, error=%s", selected_model, use_case, str(e))
            raise LLMException(f"LLM call failed: {str(e)}")

    def call_comprehensive(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """Call comprehensive model for complex reasoning tasks."""
        return self.call_llm(prompt, use_case="comprehensive", max_tokens=max_tokens)

    def call_personalization(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """Call efficient model for response personalization."""
        return self.call_llm(prompt, use_case="personalization", max_tokens=max_tokens)

    # Async versions for better performance in async contexts
    async def acall_llm(self, prompt: str, model: Optional[str] = None, max_tokens: int = 200, use_case: Optional[str] = None) -> Dict[str, Any]:
        """
        Async version of call_llm for better performance in async contexts.

        Args:
            prompt: Input prompt for the model
            model: Specific model to use (overrides use_case)
            max_tokens: Maximum tokens in response
            use_case: Use case type ("comprehensive", "personalization", "efficient", "fallback", "lightweight")

        Returns:
            Dictionary with response and metadata
        """
        # Model selection priority: explicit model > use_case > default comprehensive
        if model:
            selected_model = model
        elif use_case and use_case in self.models:
            selected_model = self.models[use_case]
        else:
            selected_model = self.models["comprehensive"]
        start_time = time.time()

        try:
            response = await self.async_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )

            latency = (time.time() - start_time) * 1000  # Convert to milliseconds

            output = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Use TokenCounter for accurate cost calculation
            cost = self.token_counter.calculate_cost(input_tokens, output_tokens, selected_model)

            model_logger.info(
                "Async LLM call completed: model=%s, use_case=%s, latency_ms=%s, input_tokens=%s, output_tokens=%s, cost_usd=%s",
                selected_model, use_case, round(latency, 2), input_tokens, output_tokens, cost
            )

            return {
                "response": output,
                "latency_ms": round(latency, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model": selected_model,
                "use_case": use_case,
                "cost_usd": cost
            }

        except Exception as e:
            model_logger.error(
                "Async LLM call failed: model=%s, use_case=%s, error=%s",
                selected_model, use_case, str(e)
            )
            raise LLMException(f"Async LLM call failed: {str(e)}")

    async def acall_comprehensive(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """Async call comprehensive model for complex reasoning tasks."""
        return await self.acall_llm(prompt, use_case="comprehensive", max_tokens=max_tokens)

    async def acall_personalization(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """Async call efficient model for response personalization."""
        return await self.acall_llm(prompt, use_case="personalization", max_tokens=max_tokens)

    async def personalize_response(
        self,
        cached_response: str,
        user_context: Dict[str, Any],
        original_prompt: str,
        max_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Personalize cached response using user context.

        Args:
            cached_response: Original cached response to personalize
            user_context: User-specific context and preferences
            original_prompt: Original user query
            max_tokens: Maximum tokens for personalized response

        Returns:
            Dictionary with personalized response and metadata
        """
        context_prompt = self._build_context_prompt(cached_response, user_context, original_prompt)
        start_time = time.time()

        try:
            # Use personalization use case model with async client
            personalization_model = self.models["personalization"]

            response = await self.async_client.chat.completions.create(
                model=personalization_model,
                messages=[
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": "Please personalize this cached response for the user. Keep your response under 5 sentences."}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )

            latency = (time.time() - start_time) * 1000
            personalized_response = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Use TokenCounter for accurate cost calculation
            cost = self.token_counter.calculate_cost(input_tokens, output_tokens, personalization_model)

            model_logger.info(
                "Response personalization completed: model=%s, use_case=%s, latency_ms=%s, input_tokens=%s, output_tokens=%s, cost_usd=%s",
                personalization_model, "personalization", round(latency, 2), input_tokens, output_tokens, cost
            )

            return {
                "response": personalized_response,
                "latency_ms": round(latency, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model": personalization_model,
                "use_case": "personalization",
                "cost_usd": cost
            }

        except Exception as e:
            model_logger.error("Response personalization failed: %s", str(e))
            raise LLMException(f"Response personalization failed: {str(e)}")

    async def chat_completion(
        self,
        request: ChatRequest,
        context: Optional[List[Dict[str, str]]] = None
    ) -> ChatResponse:
        """
        Generate chat completion response using async client.

        Args:
            request: Chat request with query and parameters
            context: Optional conversation context

        Returns:
            ChatResponse with generated response and metadata
        """
        start_time = time.time()

        try:
            messages = self._build_messages(request.query, context)
            model = settings.openai_model
            max_tokens = request.max_tokens or settings.openai_max_tokens

            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature or settings.openai_temperature,
                max_tokens=max_tokens
            )

            latency = (time.time() - start_time) * 1000
            output = response.choices[0].message.content

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Calculate cost for tracking using token_counter
            cost = self.token_counter.calculate_cost(input_tokens, output_tokens, model)

            return ChatResponse(
                response=output,
                sources=[],  # To be populated by RAG service
                cached=False,
                token_usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                latency_ms=round(latency, 2),
                metadata={
                    "model": model,
                    "cost_usd": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            )

        except Exception as e:
            model_logger.error("Async chat completion failed: %s", str(e))
            raise LLMException(f"Chat completion failed: {str(e)}")

    async def chat_completion_stream(
        self,
        request: ChatRequest,
        context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming chat completion response.

        Args:
            request: Chat request with query and parameters
            context: Optional conversation context

        Yields:
            Chunks of the response as they're generated
        """
        try:
            messages = self._build_messages(request.query, context)
            model = settings.openai_model
            max_tokens = request.max_tokens or settings.openai_max_tokens

            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature or settings.openai_temperature,
                max_tokens=max_tokens,
                stream=True
            )

            async for chunk in stream:
                try:
                    # Safely access chunk content with proper validation
                    if (chunk.choices and
                        len(chunk.choices) > 0 and
                        chunk.choices[0].delta and
                        chunk.choices[0].delta.content):
                        yield chunk.choices[0].delta.content
                except (AttributeError, IndexError) as e:
                    model_logger.warning("Failed to extract content from streaming chunk: %s", str(e))
                    continue  # Skip this chunk and continue with next one

        except Exception as e:
            model_logger.error("Streaming chat completion failed: %s", str(e))
            raise LLMException(f"Streaming chat completion failed: {str(e)}")

    def _build_messages(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build message list for OpenAI API.

        Args:
            query: User query
            context: Conversation context
            system_prompt: Optional system prompt

        Returns:
            List of messages in OpenAI format
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation context if provided
        if context:
            # Limit context to last 10 messages to avoid token limits
            limited_context = context[-10:]
            messages.extend(limited_context)

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    def _build_context_prompt(
        self,
        cached_response: str,
        user_context: Dict[str, Any],
        prompt: str
    ) -> str:
        """
        Build personalization prompt with user context.

        Args:
            cached_response: Original cached response
            user_context: User-specific context
            prompt: Original user query

        Returns:
            Personalization system prompt
        """
        context_parts = []

        if user_context.get("preferences"):
            context_parts.append("User preferences: " + ", ".join(user_context["preferences"]))

        if user_context.get("goals"):
            context_parts.append("User goals: " + ", ".join(user_context["goals"]))

        if user_context.get("history"):
            context_parts.append("User history: " + ", ".join(user_context["history"]))

        context_blob = "\n".join(context_parts)

        return f"""You are a personalization assistant. A cached response was previously generated for the prompt: "{prompt}".

Here is the cached response:
\"\"\"{cached_response}\"\"\"

Use the user's context below to personalize and refine the response:
{context_blob}

Respond in a way that feels tailored to this user, adjusting tone, content, or suggestions as needed. Keep your response under 3 sentences no matter what."""

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get current model configuration for monitoring and debugging.

        Returns:
            Dictionary with model configuration info
        """
        return {
            "available_models": self.models,
            "default_models": {
                "comprehensive": self.models["comprehensive"],
                "personalization": self.models["personalization"]
            },
            "token_counter_model": self.token_counter.model_name,
            "client_initialized": bool(self.client),
            "async_client_initialized": bool(self.async_client)
        }

    async def health_check(self) -> bool:
        """
        Check if LLM service is healthy using cost-effective model.

        Returns:
            True if service is healthy
        """
        try:
            # Use the most cost-effective model for health check
            health_check_model = self.models.get("comprehensive", self.models.get("personalization"))

            response = await self.async_client.chat.completions.create(
                model=health_check_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            model_logger.error("LLM health check failed: %s", str(e))
            return False