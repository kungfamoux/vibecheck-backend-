"""
Content management endpoints for the recommendation system.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from firebase_admin.auth import UserRecord
from typing import List, Optional, Dict, Any

from models import schemas
from services.content_service import content_service
from auth.auth_bearer import auth_scheme
from api.endpoints import get_current_user

router = APIRouter(prefix="/api/content", tags=["content"])

@router.post("/", response_model=Dict[str, Any])
async def create_content(
    content: schemas.ContentCreate,
    current_user: UserRecord = Depends(get_current_user)
):
    """
    Create new content in the recommendation system.
    
    Required fields:
    - title: Content title
    - description: Content description
    
    Optional fields:
    - category: Content category (default: "general")
    - tags: List of descriptive tags
    - mood_tags: List of mood-related tags (e.g., ["happy", "energetic"])
    - metadata: Additional content metadata as a dictionary
    """
    try:
        result = await content_service.add_content(
            title=content.title,
            description=content.description,
            category=content.category,
            tags=content.tags,
            mood_tags=content.mood_tags,
            metadata=content.metadata
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
            
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create content: {str(e)}"
        )

@router.get("/{content_id}", response_model=Dict[str, Any])
async def get_content(
    content_id: str,
    current_user: UserRecord = Depends(get_current_user)
):
    """
    Get content by ID.
    """
    try:
        content = await content_service.get_content(content_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        return content
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get content: {str(e)}"
        )

@router.patch("/{content_id}", response_model=Dict[str, Any])
async def update_content(
    content_id: str,
    update_data: Dict[str, Any],
    current_user: UserRecord = Depends(get_current_user)
):
    """
    Update existing content.
    
    Note: Cannot update content_id or created_at fields.
    """
    try:
        result = await content_service.update_content(content_id, update_data)
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update content: {str(e)}"
        )
