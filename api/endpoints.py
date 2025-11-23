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

@router.post("/recommendations", response_model=List[schemas.RecommendationResponse])
async def get_recommendations(
    query: schemas.RecommendationQuery
):
    """
    Get content recommendations based on input text and preferences from social media platforms
    
    Request body should contain:
    - **user_text**: The input text to analyze for mood and preferences
    - **limit**: (Optional) Maximum number of recommendations to return (default: 10, max: 20)
    - **content_types**: (Optional) List of content types to include (e.g., ['video', 'playlist', 'track'])
    - **platforms**: (Optional) List of platforms to include (e.g., ['youtube', 'spotify'])
    
    Returns a list of recommended content items with match scores from social media platforms
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
        
        # Set default values if not provided
        limit = min(20, int(query.limit)) if hasattr(query, 'limit') and query.limit else 10
        
        logger.info(f"Detected mood: {primary_mood}, fetching recommendations...")
        
        # Get recommendations based on mood from social media platforms
        recommendations = await recommendation_engine.get_similar_mood_content(
            mood=primary_mood,
            top_n=limit
        )
        
        if not recommendations:
            # Fallback to general recommendations if no mood-specific content found
            recommendations = await recommendation_engine.get_similar_mood_content(
                mood="motivation",
                top_n=limit
            )
        
        if not recommendations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No matching content found. Please try a different mood or try again later."
            )
        
        # Apply content type and platform filters if specified
        filtered_recommendations = []
        for item in recommendations:
            try:
                # Skip if content type filter is specified and doesn't match
                if hasattr(query, 'content_types') and query.content_types:
                    item_type = item.get('type', '').lower()
                    if item_type not in [t.lower() for t in query.content_types]:
                        continue
                
                # Skip if platform filter is specified and doesn't match
                if hasattr(query, 'platforms') and query.platforms:
                    item_source = item.get('source', '').lower()
                    if item_source not in [p.lower() for p in query.platforms]:
                        continue
                
                filtered_recommendations.append(item)
            except Exception as e:
                logger.error(f"Error filtering recommendation: {e}")
                continue
        
        # If no items match the filters, return the original recommendations
        if not filtered_recommendations and (hasattr(query, 'content_types') or hasattr(query, 'platforms')):
            filtered_recommendations = recommendations
        
        # Format the response
        formatted_recommendations = []
        for item in filtered_recommendations[:limit]:  # Ensure we don't exceed the limit
            try:
                # Calculate match score percentage (0-100%)
                match_score = min(100, max(0, int(item.get('match_score', 0.5) * 100)))
                
                # Base item structure
                formatted_item = {
                    'id': item.get('id', ''),
                    'title': item.get('title', 'Untitled'),
                    'description': item.get('description', ''),
                    'match_score': match_score,
                    'source': item.get('source', 'external'),
                    'url': item.get('url', ''),
                    'thumbnail': item.get('thumbnail') or item.get('image_url', ''),
                    'type': item.get('type', 'content'),
                    'mood': item.get('mood', primary_mood),
                    'tags': item.get('tags', []),
                    'created_at': item.get('created_at', datetime.now(timezone.utc).isoformat())
                }
                
                # Add platform-specific metadata
                if item.get('source') == 'youtube':
                    formatted_item.update({
                        'duration': item.get('duration'),
                        'view_count': item.get('view_count'),
                        'channel': item.get('channel')
                    })
                elif item.get('source') == 'spotify':
                    formatted_item.update({
                        'artists': item.get('artists', []),
                        'album': item.get('album'),
                        'duration_ms': item.get('duration_ms'),
                        'preview_url': item.get('preview_url')
                    })
                    if item.get('type') == 'playlist':
                        formatted_item['track_count'] = item.get('track_count')
                
                formatted_recommendations.append(formatted_item)
                
            except Exception as e:
                logger.error(f"Error formatting recommendation: {e}")
                continue
        
        # Sort by match score (highest first) and return
        formatted_recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return formatted_recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}", exc_info=True)
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
