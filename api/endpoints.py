from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Optional
from pydantic import BaseModel
import firebase_admin
from firebase_admin import auth
from firebase_admin.auth import UserRecord

from models import schemas
from services import ml_service, recommendation_engine
from auth.auth_bearer import auth_scheme
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])

# Helper function to get the current user from the token
async def get_current_user(token: str = Depends(auth_scheme)) -> UserRecord:
    try:
        # The token is already verified by JWTBearer, now get the user
        user = auth.get_user(token)
        return user
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/auth/login", response_model=schemas.Token)
async def login(login_data: dict = Body(...)):
    """
    Login with Firebase ID token
    
    Request body should contain:
    - id_token: The Firebase ID token from the client
    
    Returns:
    - access_token: JWT token for authenticated requests
    - token_type: Always "bearer"
    """
    try:
        id_token = login_data.get("id_token")
        if not id_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID token is required"
            )
        
        # Verify the ID token
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        
        # Get the user's Firebase UID
        user = auth.get_user(uid)
        
        # In a real app, you might want to create/update the user in your database here
        
        # Return the ID token as the access token
        return {
            "access_token": id_token,
            "token_type": "bearer"
        }
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/content/{content_id}/comment", response_model=schemas.Sentiment)
async def add_comment(
    content_id: str,
    comment_data: schemas.SentimentCreate,
    current_user: UserRecord = Depends(get_current_user)
):
    """
    Add a comment to content and analyze its sentiment
    
    - **content_id**: ID of the content to comment on
    - **comment_data**: The comment text and metadata
    
    Returns the created comment with sentiment analysis
    """
    try:
        from models.database import db
        from datetime import datetime
        
        # Analyze the sentiment of the comment
        sentiment_analysis = ml_service.analyze_content_feedback(comment_data.user_comment)
        
        # Create the sentiment document
        sentiment_data = {
            "user_id": current_user.uid,
            "content_id": content_id,
            "user_comment": comment_data.user_comment,
            "sentiment_score": sentiment_analysis["sentiment_score"],
            "sentiment_label": sentiment_analysis["sentiment_label"],
            "created_at": datetime.utcnow()
        }
        
        # Save to Firestore
        doc_ref = db.collection("sentiment").document()
        doc_ref.set(sentiment_data)
        
        # Return the created sentiment with the generated ID
        sentiment_data["sentiment_id"] = doc_ref.id
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error adding comment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add comment"
        )

@router.post("/recommendations/query", response_model=List[schemas.RecommendationResponse])
async def get_recommendations(
    query: schemas.RecommendationQuery,
    current_user: UserRecord = Depends(get_current_user)
):
    """
    Get content recommendations based on user input text
    
    - **user_text**: The user's input text to analyze for mood and preferences
    
    Returns a list of recommended content items with match scores
    """
    try:
        # Analyze the user's text to get mood tags
        mood_tags_with_scores = ml_service.predict_content_tags(query.user_text)
        mood_tags = [tag for tag, score in mood_tags_with_scores]
        
        if not mood_tags:
            # Default to some general tags if no specific mood is detected
            mood_tags = ["general", "popular"]
        
        logger.info(f"Detected mood tags: {mood_tags}")
        
        # Get recommendations based on mood tags
        recommendations = await recommendation_engine.get_recommendations(
            mood_tags=mood_tags,
            limit=10,
            user_id=current_user.uid
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
