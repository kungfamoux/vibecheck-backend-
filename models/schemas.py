from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from datetime import datetime
from enum import Enum

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str
    
class UserLogin(BaseModel):
    email: str
    password: str
    
class UserResponse(UserBase):
    uid: str
    created_at: str
    is_active: bool = True

class User(UserBase):
    user_id: str
    registration_date: datetime

    class Config:
        from_attributes = True

class ContentItem(BaseModel):
    content_id: str
    title: str
    description: Optional[str] = None
    category: str = "general"
    tags: List[str] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = {}

    class Config:
        from_attributes = True

class ContentBase(BaseModel):
    title: str
    description: str
    category: str
    mood_tags: List[str]

class ContentCreate(ContentBase):
    pass

class Content(ContentBase):
    content_id: str
    created_at: datetime

    class Config:
        from_attributes = True

class InteractionType(str, Enum):
    VIEW = "view"
    RATING = "rating"

class InteractionBase(BaseModel):
    user_id: str
    content_id: str
    interaction_type: InteractionType
    metadata: Optional[dict] = {}

class InteractionCreate(InteractionBase):
    pass

class Interaction(InteractionBase):
    interaction_id: str
    timestamp: datetime

    class Config:
        from_attributes = True

class RecommendationQuery(BaseModel):
    user_text: str = Field(..., description="User's input text for content recommendation")
    limit: Optional[int] = Field(10, description="Maximum number of recommendations to return")
    include_external: Optional[bool] = Field(True, description="Whether to include content from external APIs")
    min_sentiment: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum sentiment score (0-1)")
    max_sentiment: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Maximum sentiment score (0-1)")
    content_types: Optional[List[str]] = Field(None, description="Filter by content types (e.g., ['video', 'article'])")

class UserInteraction(BaseModel):
    """Represents a user's interaction with content"""
    user_id: str
    content_id: str
    interaction_type: InteractionType
    timestamp: datetime
    metadata: Optional[dict] = {}

    class Config:
        from_attributes = True

class RecommendationResponse(BaseModel):
    content_id: str
    title: str
    description: str
    category: str = "general"
    match_score: float
    source: str = "database"
    url: Optional[str] = None
    thumbnail: Optional[str] = None
    type: str = "article"
    sentiment: float = 0.5
    tags: List[str] = []
    created_at: Optional[str] = None
    duration: Optional[str] = None
    track_count: Optional[int] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class MoodAnalysis(BaseModel):
    """Schema for mood analysis results"""
    primary_mood: str
    confidence: float
    sentiment: str
    mood_breakdown: dict
    mood_transitions: Optional[List[dict]] = None

class TextMoodRequest(BaseModel):
    """Schema for text mood analysis request"""
    text: str
    analyze_transitions: bool = False
