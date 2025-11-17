from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class UserContext(BaseModel):
    """Model for user context for personalization."""
    user_id: str = Field(..., description="User ID")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Recent conversation history")
    interests: List[str] = Field(default_factory=list, description="User interests/topics")
    expertise_level: str = Field(default="intermediate", description="User expertise level")
    learning_style: Optional[str] = Field(None, description="Preferred learning style")
    language_preference: str = Field(default="en", description="Language preference")
    session_data: Dict[str, Any] = Field(default_factory=dict, description="Session-specific data")
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Additional fields for ChatService integration
    preferences: List[str] = Field(default_factory=list, description="User preferences")
    goals: List[str] = Field(default_factory=list, description="User goals")
    history: List[str] = Field(default_factory=list, description="User interaction history")

