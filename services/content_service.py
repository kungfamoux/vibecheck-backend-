"""
Content management service for handling content operations in the recommendation system.
"""
from datetime import datetime
from typing import Dict, List, Optional
import uuid
from models.database import content_ref
from models.schemas import ContentCreate
import logging

logger = logging.getLogger(__name__)

class ContentService:
    """Service for managing content in the recommendation system."""
    
    @staticmethod
    async def add_content(
        title: str,
        description: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
        mood_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Add new content to the recommendation system.
        
        Args:
            title: Content title
            description: Content description
            category: Content category (e.g., "technology", "entertainment")
            tags: List of descriptive tags
            mood_tags: List of mood-related tags (e.g., ["happy", "energetic"])
            metadata: Additional content metadata
            
        Returns:
            Dictionary with the created content data
        """
        try:
            # Generate a unique ID for the content
            content_id = str(uuid.uuid4())
            current_time = datetime.utcnow()
            
            # Prepare content data
            content_data = {
                "content_id": content_id,
                "title": title,
                "description": description,
                "category": category.lower(),
                "tags": tags or [],
                "mood_tags": mood_tags or [],
                "created_at": current_time,
                "updated_at": current_time,
                "metadata": metadata or {},
                "interaction_count": 0,
                "popularity": 0.0
            }
            
            # Add to Firestore
            doc_ref = content_ref.document(content_id)
            doc_ref.set(content_data)
            
            logger.info(f"Successfully added content: {content_id}")
            return {
                "status": "success",
                "content_id": content_id,
                "data": content_data
            }
            
        except Exception as e:
            logger.error(f"Failed to add content: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to add content: {str(e)}"
            }
    
    @staticmethod
    async def get_content(content_id: str) -> Optional[Dict]:
        """Get content by ID."""
        try:
            doc = content_ref.document(content_id).get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting content {content_id}: {str(e)}")
            return None
    
    @staticmethod
    async def update_content(
        content_id: str,
        update_data: Dict
    ) -> Dict:
        """Update existing content."""
        try:
            if not update_data:
                return {"status": "error", "message": "No update data provided"}
                
            # Don't allow updating these fields
            update_data.pop('content_id', None)
            update_data.pop('created_at', None)
            
            # Set updated_at timestamp
            update_data['updated_at'] = datetime.utcnow()
            
            doc_ref = content_ref.document(content_id)
            doc_ref.update(update_data)
            
            logger.info(f"Successfully updated content: {content_id}")
            return {"status": "success", "content_id": content_id}
            
        except Exception as e:
            logger.error(f"Error updating content {content_id}: {str(e)}")
            return {"status": "error", "message": f"Failed to update content: {str(e)}"}

# Create a global instance
content_service = ContentService()
