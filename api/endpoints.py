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

@router.post("/contents/{content_id}/comments", response_model=schemas.Sentiment, status_code=status.HTTP_201_CREATED)
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

@router.post("/recommendations", response_model=List[schemas.RecommendationResponse])
async def get_recommendations(
    query: schemas.RecommendationQuery,
    current_user: UserRecord = Depends(get_current_user)
):
    """
    Get personalized content recommendations based on user input text and preferences
    
    Request body should contain:
    - **user_text**: The user's input text to analyze for mood and preferences
    
    Returns a list of recommended content items with match scores
    """
    try:
        if not query.user_text or not query.user_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_text is required in the request body"
            )
            
        logger.info(f"Getting recommendations for user {current_user.uid} based on text: {query.user_text[:100]}...")
        
        # Analyze the user's text to get mood tags
        mood_tags_with_scores = ml_service.predict_content_tags(query.user_text)
        mood_tags = [tag for tag, score in mood_tags_with_scores if score > 0.3]  # Filter out low confidence tags
        
        if not mood_tags:
            # Default to some general tags if no specific mood is detected
            mood_tags = ["general", "popular"]
        
        logger.info(f"Detected mood tags: {mood_tags}")
        
        # Get recommendations based on mood tags
        recommendations = await recommendation_engine.get_recommendations(
            mood_tags=mood_tags,
            limit=10,
            user_id=current_user.uid,
            user_text=query.user_text
        )
        
        # Log the recommendation request for future personalization
        try:
            from models.database import db
            from datetime import datetime
            
            db.collection("recommendation_requests").add({
                "user_id": current_user.uid,
                "user_text": query.user_text,
                "mood_tags": mood_tags,
                "recommendation_count": len(recommendations),
                "created_at": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to log recommendation request: {e}")
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating recommendations"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
