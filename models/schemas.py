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
    COMMENT = "comment"

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

class SentimentBase(BaseModel):
    user_id: str
    content_id: str
    user_comment: str

class SentimentCreate(SentimentBase):
    pass

class Sentiment(SentimentBase):
    sentiment_id: str
    sentiment_score: float
    created_at: datetime

    class Config:
        from_attributes = True

class RecommendationQuery(BaseModel):
    user_text: str = Field(..., description="User's input text for content recommendation")

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
    category: str
    match_score: float

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
