# Tutor Services - RAG Chatbot with Semantic Caching

A FastAPI-based service providing RAG (Retrieval-Augmented Generation) chatbot capabilities with semantic caching using Redis vector store.

## Features

- **RAG Pipeline**: Document retrieval and context-aware generation
- **Semantic Caching**: Intelligent caching with context personalization
- **FastAPI REST API**: Modern async API with comprehensive endpoints
- **Telemetry**: Cost and latency tracking for optimization
- **Redis Vector Store**: High-performance vector similarity search
- **OpenAI Integration**: GPT models and text embeddings

## Architecture

```
tutor-services/
├── app/
│   ├── core/          # LLM client, embeddings, telemetry
│   ├── services/      # Semantic cache, RAG service, chat service
│   ├── repositories/  # Redis operations, vector store, cache
│   ├── models/        # Pydantic models for all entities
│   ├── api/           # FastAPI routes and endpoints
│   └── config/        # Settings and configuration
├── tests/             # Test suite
├── main.py           # FastAPI application entry point
├── requirements.txt  # Python dependencies
└── .env.example      # Environment variables template
```

## Prerequisites

- Python 3.9+
- Redis 6.0+ with RedisJSON and RedisSearch modules
- OpenAI API key

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd tutor-services
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Unix/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` file with your configuration:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
SECRET_KEY=your_secret_key_here

# Redis (default: redis://localhost:6379)
REDIS_URL=redis://localhost:6379

# OpenAI Models
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### 5. Start Redis

Using Docker (recommended):

```bash
docker run -d --name redis-stack \
  -p 6379:6379 \
  redis/redis-stack-server:latest
```

Or install Redis locally with RedisJSON and RedisSearch modules.

### 6. Run the application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the service is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Chat Endpoints

- `POST /chat/completion` - Generate chat response
- `POST /chat/completion/stream` - Streaming chat response
- `GET /chat/history/{user_id}` - Get conversation history
- `DELETE /chat/history/{user_id}` - Clear conversation history
- `PUT /chat/context/{user_id}` - Update user context

### Health Endpoints

- `GET /health/` - Overall health check
- `GET /health/components` - Component health status
- `GET /health/readiness` - Readiness probe
- `GET /health/liveness` - Liveness probe

### Admin Endpoints

- `GET /admin/stats` - System statistics
- `GET /admin/cache/stats` - Cache statistics
- `DELETE /admin/cache` - Clear cache
- `POST /admin/cache/optimize` - Optimize cache
- `GET /admin/documents/stats` - Document statistics
- `POST /admin/documents/reindex` - Re-index documents
- `GET /admin/telemetry/metrics` - Telemetry metrics
- `DELETE /admin/telemetry/cleanup` - Cleanup telemetry data

## Usage Examples

### Basic Chat Completion

```python
import httpx

async def chat_example():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat/completion",
            json={
                "query": "What is machine learning?",
                "user_id": "user123"
            }
        )
        result = response.json()
        print(result["response"])
```

### Streaming Chat

```python
import httpx

async def streaming_chat_example():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/chat/completion/stream",
            json={"query": "Explain quantum computing"}
        ) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="")
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

### Code Formatting

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/
```

### Type Checking

```bash
mypy app/
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model | `gpt-3.5-turbo` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `CACHE_THRESHOLD` | Semantic similarity threshold | `0.85` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `RAG_TOP_K` | RAG retrieval count | `5` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Monitoring and Telemetry

The service includes comprehensive telemetry tracking:

- **Cost Tracking**: Monitor OpenAI API costs
- **Latency Monitoring**: Track response times
- **Cache Analytics**: Hit rates and optimization
- **Usage Metrics**: Request counts and patterns

Access metrics via:
- Admin API endpoints
- Structured logs (JSON format)
- Custom monitoring integration

## Performance Optimization

### Semantic Caching

- Automatically caches similar queries
- Personalized responses per user
- Configurable similarity thresholds
- TTL-based expiration

### RAG Optimization

- Vector similarity search with Redis
- Configurable chunk sizes
- Batch processing for documents
- Result reranking options

### Redis Configuration

For optimal performance:

```redis
# Redis configuration recommendations
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Ensure Redis is running with required modules
   - Check Redis URL configuration
   - Verify network connectivity

2. **OpenAI API Errors**
   - Verify API key is valid
   - Check rate limits and quotas
   - Ensure correct model names

3. **Vector Index Issues**
   - Rebuild index: `POST /admin/documents/reindex`
   - Check Redis modules are loaded
   - Verify vector dimensions

### Health Checks

Monitor service health:

```bash
curl http://localhost:8000/health/
curl http://localhost:8000/health/components
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run test suite
5. Submit pull request

## License

[License Information]

## Support

For support and questions:
- Create GitHub issue
- Check API documentation
- Review troubleshooting guide"# lms-tutor-ai" 
