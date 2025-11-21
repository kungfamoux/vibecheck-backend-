from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import firebase_admin
from firebase_admin import auth
from firebase_admin.auth import UserRecord

from models import schemas
from services.ml_service import ml_service
from services.recommendation_engine import MoodBasedRecommender

# Create an instance of the recommendation engine
recommendation_engine = MoodBasedRecommender()

# Initialize sample content on startup
import asyncio
asyncio.create_task(recommendation_engine.initialize_sample_content())

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

@router.post("/contents/{content_id}/comments", response_model=schemas.Sentiment, status_code=status.HTTP_201_CREATED)
async def add_comment(
    content_id: str,
    comment_data: schemas.SentimentCreate
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
        
        # Verify content exists
        content_ref = db.collection("contents").document(content_id)
        if not content_ref.get().exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content with ID {content_id} not found"
            )
        
        # Analyze the sentiment of the comment
        sentiment_analysis = ml_service.analyze_content_feedback(comment_data.user_comment)
        
        # Create the comment document
        comment_data = {
            "user_id": current_user.uid,
            "user_email": current_user.email,
            "content_id": content_id,
            "comment": comment_data.user_comment,
            "sentiment_score": float(sentiment_analysis.get("sentiment_score", 0.0)),
            "sentiment_label": sentiment_analysis.get("sentiment_label", "neutral"),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Save to Firestore
        doc_ref = db.collection("comments").document()
        doc_ref.set(comment_data)
        
        # Update content's comment count
        content_ref.update({
            "comment_count": firestore.Increment(1),
            "updated_at": datetime.utcnow().isoformat()
        })
        
        # Return the created comment with the generated ID
        return {
            "comment_id": doc_ref.id,
            "content_id": content_id,
            "user_id": current_user.uid,
            "comment": comment_data["comment"],
            "sentiment_score": comment_data["sentiment_score"],
            "sentiment_label": comment_data["sentiment_label"],
            "created_at": comment_data["created_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding comment: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while adding the comment"
        )

@router.post("/init-content", status_code=status.HTTP_200_OK)
async def initialize_sample_content():
    """
    Initialize sample content in the database.
    This endpoint is for development purposes only.
    """
    try:
        success = await recommendation_engine.initialize_sample_content()
        if success:
            return {"status": "success", "message": "Sample content initialized successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize sample content"
            )
    except Exception as e:
        logger.error(f"Error initializing sample content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/recommendations", response_model=List[schemas.RecommendationResponse])
async def get_recommendations(
    query: schemas.RecommendationQuery
):
    """
    Get content recommendations based on input text and preferences
    
    Request body should contain:
    - **user_text**: The input text to analyze for mood and preferences
    
    Returns a list of recommended content items with match scores
    """
    try:
        if not query.user_text or not query.user_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_text is required in the request body"
            )
            
        logger.info(f"Getting recommendations based on text: {query.user_text[:100]}...")
        
        # Analyze the user's text to get mood
        mood_result = ml_service.detect_mood(query.user_text)
        primary_mood = mood_result.get('primary_mood', 'neutral')
        mood_tags = [primary_mood]
        
        if not mood_tags:
            # Default to some general tags if no specific mood is detected
            mood_tags = ["general", "popular"]
        
        logger.info(f"Detected mood: {primary_mood}")
        
        # Ensure we have content loaded
        await recommendation_engine._refresh_content_cache()
        
        if not recommendation_engine.content_cache:
            logger.warning("No content available in the database")
            # Try to initialize sample content if none exists
            await recommendation_engine.initialize_sample_content()
            await recommendation_engine._refresh_content_cache()
            
            if not recommendation_engine.content_cache:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No content available for recommendations"
                )
        
        # Get all content and score based on mood match
        all_content = list(recommendation_engine.content_cache.values())
        
        # Score each content item based on mood match
        scored_content = []
        for content in all_content:
            content_text = f"{content.get('title', '')} {content.get('description', '')}"
            content_mood = ml_service.detect_mood(content_text)
            content_primary_mood = content_mood.get('primary_mood', 'neutral')
            
            # Calculate match score (simple for now, can be enhanced)
            mood_match = 1.0 if content_primary_mood == primary_mood else 0.5
            
            # Include sentiment in scoring
            sentiment_score = content.get('sentiment', 0.5)  # Default to neutral if not set
            
            # Combine scores (weighted average)
            score = (mood_match * 0.7) + (sentiment_score * 0.3)
            
            scored_content.append({
                'id': content.get('id', ''),
                'title': content.get('title', 'Untitled'),
                'description': content.get('description', ''),
                'mood': content_primary_mood,
                'sentiment': 'positive' if sentiment_score > 0.6 else 'negative' if sentiment_score < 0.4 else 'neutral',
                'score': min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
            })
        
        # Sort by score (highest first) and take top 10
        recommendations = sorted(
            scored_content, 
            key=lambda x: x['score'], 
            reverse=True
        )[:10]
        
        # Log the recommendation request
        try:
            from models.database import db
            from datetime import datetime
            
            db.reference('recommendation_requests').push({
                "user_text": query.user_text,
                "mood_tags": mood_tags,
                "primary_mood": primary_mood,
                "recommendation_count": len(recommendations),
                "is_anonymous": True,
                "created_at": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to log recommendation request: {e}")
        
        return recommendations
        
    except HTTPException as he:
        logger.error(f"HTTP error in get_recommendations: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating recommendations: {str(e)}"
        )

@router.post("/analyze/mood", response_model=schemas.MoodAnalysis)
async def analyze_mood(
    request: schemas.TextMoodRequest
):
    """
    Analyze text to detect mood and sentiment
    
    - **text**: The input text to analyze
    - **analyze_transitions**: Whether to analyze mood changes in the text (for longer texts)
    
    Returns mood analysis including:
    - primary_mood: The most dominant mood
    - confidence: Confidence score (0-1)
    - sentiment: Overall sentiment
    - mood_breakdown: All detected moods with scores
    - mood_transitions: (if analyze_transitions=True) Mood changes in the text
    """
    try:
        # Use the ML service to detect mood
        mood_analysis = ml_service.detect_mood(
            text=request.text,
            analyze_transitions=request.analyze_transitions
        )
        
        # Convert the mood analysis to the response model
        return {
            "primary_mood": mood_analysis.get("primary_mood", "neutral"),
            "confidence": mood_analysis.get("confidence", 0.0),
            "sentiment": mood_analysis.get("sentiment", "neutral"),
            "mood_breakdown": mood_analysis.get("mood_breakdown", {}),
            "mood_transitions": mood_analysis.get("mood_transitions", []) if request.analyze_transitions else None
        }
        
    except Exception as e:
        logger.error(f"Error analyzing mood: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error analyzing mood"
        )

# Health check endpoint
@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}
