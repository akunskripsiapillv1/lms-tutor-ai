"""
Tutor Services - RAG Chatbot with Semantic Caching

A FastAPI-based service providing RAG (Retrieval-Augmented Generation)
chatbot capabilities with semantic caching using Redis vector store.
"""

__version__ = "0.1.0"
__author__ = "Tutor Services Team"

from .config.settings import settings
from .api import chat_router, health_router

__all__ = [
    "settings",
    "chat_router",
    "health_router"
]