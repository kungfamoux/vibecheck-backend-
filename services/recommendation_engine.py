from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import json
import os

# Import ML service
from .ml_service import ml_service
from models.database import content_ref, interactions_ref, user_profiles_ref, get_sentiment_rtd
from models.schemas import RecommendationResponse, ContentItem, UserInteraction

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Recommendation Engine that provides content recommendations based on:
    1. Content-based filtering
    2. Collaborative filtering
    3. Sentiment analysis
    4. User preferences
    """
    
    def __init__(self):
        # Content cache
        self.content_cache = {}
        self.content_embeddings = {}
        self.last_cache_update = None
        self.cache_ttl = 3600  # 1 hour cache TTL
        
        # User interaction data
        self.user_interactions = {}
        self.user_profiles = {}
        
        # Initialize TF-IDF vectorizer for content analysis
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        logger.info("RecommendationEngine initialized")
    
    async def _refresh_content_cache(self):
        """Refresh the content cache if it's stale."""
        current_time = datetime.utcnow()
        
        if (self.last_cache_update is None or 
            (current_time - self.last_cache_update).total_seconds() > self.cache_ttl):
            
            logger.info("Refreshing content cache...")
            try:
                # Fetch all content from Firestore
                content_docs = content_ref.stream()
                
                # Convert to dictionary with content_id as key
                self.content_cache = {
                    doc.id: {**doc.to_dict(), 'content_id': doc.id} 
                    for doc in content_docs
                }
                
                # Precompute content embeddings for similarity
                await self._compute_content_embeddings()
                
                self.last_cache_update = current_time
                logger.info(f"Content cache refreshed with {len(self.content_cache)} items.")
                
            except Exception as e:
                logger.error(f"Error refreshing content cache: {e}")
                if not self.content_cache:
                    raise
    
    async def _compute_content_embeddings(self):
        """Compute TF-IDF embeddings for all content items."""
        if not self.content_cache:
            return
            
        try:
            # Prepare text data for TF-IDF
            content_texts = []
            content_ids = []
            
            for content_id, content in self.content_cache.items():
                # Combine title, description, and tags for better representation
                text = f"{content.get('title', '')} {content.get('description', '')} {' '.join(content.get('tags', []))}"
                content_texts.append(text)
                content_ids.append(content_id)
            
            # Fit and transform the text data
            tfidf_matrix = self.vectorizer.fit_transform(content_texts)
            
            # Store the embeddings
            for i, content_id in enumerate(content_ids):
                self.content_embeddings[content_id] = tfidf_matrix[i]
                
        except Exception as e:
            logger.error(f"Error computing content embeddings: {e}")
            self.content_embeddings = {}
    
    async def _get_similar_content(self, content_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar content items using content-based filtering.
        
        Args:
            content_id: ID of the reference content
            top_n: Number of similar items to return
            
        Returns:
            List of similar content items with similarity scores
        """
        if not self.content_embeddings or content_id not in self.content_embeddings:
            return []
            
        try:
            # Get the embedding for the reference content
            ref_embedding = self.content_embeddings[content_id]
            
            # Calculate similarity with all other content
            similarities = []
            for other_id, other_embedding in self.content_embeddings.items():
                if other_id == content_id:
                    continue
                    
                # Calculate cosine similarity
                similarity = float(cosine_similarity(ref_embedding, other_embedding)[0][0])
                similarities.append((other_id, similarity))
            
            # Sort by similarity and get top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return content details with similarity scores
            return [
                {
                    **self.content_cache[content_id],
                    'similarity_score': score,
                    'content_id': content_id
                }
                for content_id, score in similarities[:top_n]
            ]
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    async def _get_content_based_recommendations(self, user_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get content-based recommendations for a user.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended content items
        """
        try:
            # Get user's interaction history
            user_history = await self._get_user_interactions(user_id)
            
            if not user_history:
                return []
            
            # Get content IDs from user history (sorted by interaction strength)
            content_scores = {}
            for interaction in user_history:
                content_id = interaction['content_id']
                # Simple scoring: higher weight for more recent interactions
                content_scores[content_id] = content_scores.get(content_id, 0) + 1
            
            # Get top interacted content
            top_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get similar content for each top interacted item
            recommendations = []
            for content_id, _ in top_content:
                similar = await self._get_similar_content(content_id, top_n=top_n)
                recommendations.extend(similar)
            
            # Sort by similarity score and remove duplicates
            seen = set()
            unique_recommendations = []
            
            for rec in sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True):
                if rec['content_id'] not in seen:
                    seen.add(rec['content_id'])
                    unique_recommendations.append(rec)
                    
                    if len(unique_recommendations) >= top_n:
                        break
            
            return unique_recommendations[:top_n]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    async def _get_collaborative_recommendations(self, user_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get collaborative filtering recommendations for a user.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended content items
        """
        # TODO: Implement collaborative filtering
        # This is a placeholder implementation
        return []
    
    async def _get_trending_content(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get trending content based on recent interactions.
        
        Args:
            top_n: Number of trending items to return
            
        Returns:
            List of trending content items
        """
        try:
            # Get recent interactions (last 7 days)
            time_threshold = datetime.utcnow() - timedelta(days=7)
            interactions = interactions_ref.where('timestamp', '>=', time_threshold).stream()
            
            # Count interactions per content
            content_counts = {}
            for doc in interactions:
                data = doc.to_dict()
                content_id = data.get('content_id')
                if content_id:
                    content_counts[content_id] = content_counts.get(content_id, 0) + 1
            
            # Sort by interaction count and get top N
            trending = sorted(content_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Get content details
            results = []
            for content_id, _ in trending:
                if content_id in self.content_cache:
                    results.append(self.content_cache[content_id])
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting trending content: {e}")
            return []
    
    async def _get_personalized_recommendations(
        self, 
        user_id: str, 
        top_n: int = 5,
        content_based_weight: float = 0.6,
        collaborative_weight: float = 0.3,
        trending_weight: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations by combining multiple strategies.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            content_based_weight: Weight for content-based recommendations
            collaborative_weight: Weight for collaborative filtering
            trending_weight: Weight for trending content
            
        Returns:
            List of recommended content items with scores
        """
        try:
            # Get recommendations from different strategies
            content_based = await self._get_content_based_recommendations(user_id, top_n * 2)
            collaborative = await self._get_collaborative_recommendations(user_id, top_n * 2)
            trending = await self._get_trending_content(top_n * 2)
            
            # Score and combine recommendations
            scored_items = {}
            
            # Add content-based recommendations
            for item in content_based:
                content_id = item['content_id']
                if content_id not in scored_items:
                    scored_items[content_id] = {
                        'content': item,
                        'score': 0.0
                    }
                scored_items[content_id]['score'] += content_based_weight * item.get('similarity_score', 0)
            
            # Add collaborative recommendations
            for item in collaborative:
                content_id = item['content_id']
                if content_id not in scored_items:
                    scored_items[content_id] = {
                        'content': item,
                        'score': 0.0
                    }
                scored_items[content_id]['score'] += collaborative_weight * item.get('score', 0.5)
            
            # Add trending content
            for item in trending:
                content_id = item['content_id']
                if content_id not in scored_items:
                    scored_items[content_id] = {
                        'content': item,
                        'score': 0.0
                    }
                scored_items[content_id]['score'] += trending_weight * 0.8  # Base score for trending
            
            # Sort by score and get top N
            sorted_items = sorted(
                scored_items.values(), 
                key=lambda x: x['score'], 
                reverse=True
            )[:top_n]
            
            # Prepare final result
            return [
                {
                    **item['content'],
                    'recommendation_score': item['score']
                }
                for item in sorted_items
            ]
            
        except Exception as e:
            logger.error(f"Error in personalized recommendations: {e}")
            return []
    
    async def get_recommendations(
        self, 
        user_id: str, 
        top_n: int = 5,
        strategy: str = 'hybrid'
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            strategy: Recommendation strategy ('content', 'collaborative', 'trending', 'hybrid')
            
        Returns:
            List of recommended content items
        """
        try:
            # Ensure content cache is up to date
            await self._refresh_content_cache()
            
            # Get recommendations based on strategy
            if strategy == 'content':
                return await self._get_content_based_recommendations(user_id, top_n)
            elif strategy == 'collaborative':
                return await self._get_collaborative_recommendations(user_id, top_n)
            elif strategy == 'trending':
                return await self._get_trending_content(top_n)
            else:  # hybrid
                return await self._get_personalized_recommendations(user_id, top_n)
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    async def _get_user_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get a user's interaction history.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of user interactions
        """
        try:
            interactions = interactions_ref.where('user_id', '==', user_id).stream()
            return [doc.to_dict() for doc in interactions]
        except Exception as e:
            logger.error(f"Error getting user interactions: {e}")
            return []
    
    async def _get_content_sentiment_scores(self) -> Dict[str, float]:
        """
        Get average sentiment scores for all content.
        
        Returns:
            Dictionary mapping content_id to average sentiment score
        """
        try:
            # Get all sentiment documents
            sentiment_docs = get_sentiment_rtd().get()
            if not sentiment_docs:
                return {}
                
            # Group by content_id and calculate average sentiment
            sentiment_scores = {}
            count = {}
            
            for doc_id, doc_data in sentiment_docs.items():
                content_id = doc_data.get('content_id')
                score = doc_data.get('sentiment_score', 0)
                
                if content_id:
                    if content_id in sentiment_scores:
                        sentiment_scores[content_id] += score
                        count[content_id] += 1
                    else:
                        sentiment_scores[content_id] = score
                        count[content_id] = 1
            
            # Calculate average sentiment scores
            avg_scores = {
                content_id: score / count[content_id]
                for content_id, score in sentiment_scores.items()
            }
            
            return avg_scores
            
        except Exception as e:
            logger.error(f"Error getting content sentiment scores: {e}")
            return {}
    
    async def get_content_by_sentiment(
        self, 
        min_sentiment: float = 0.5, 
        max_sentiment: float = 1.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get content filtered by sentiment score range.
        
        Args:
            min_sentiment: Minimum sentiment score (0-1)
            max_sentiment: Maximum sentiment score (0-1)
            limit: Maximum number of results to return
            
        Returns:
            List of content items with sentiment scores
        """
        try:
            # Get sentiment scores
            sentiment_scores = await self._get_content_sentiment_scores()
            
            # Filter by sentiment range
            filtered_content = []
            for content_id, score in sentiment_scores.items():
                if min_sentiment <= score <= max_sentiment:
                    if content_id in self.content_cache:
                        content = self.content_cache[content_id].copy()
                        content['sentiment_score'] = score
                        filtered_content.append(content)
            
            # Sort by sentiment score (descending)
            filtered_content.sort(key=lambda x: x.get('sentiment_score', 0), reverse=True)
            
            return filtered_content[:limit]
            
        except Exception as e:
            logger.error(f"Error getting content by sentiment: {e}")
            return []

# Create a global instance of the recommendation engine
recommendation_engine = RecommendationEngine()

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def test_recommendations():
        engine = RecommendationEngine()
        
        # Test content-based recommendations
        print("\nTesting content-based recommendations...")
        content_recs = await engine._get_content_based_recommendations("test_user", top_n=3)
        print(f"Content-based recommendations: {len(content_recs)} items")
        
        # Test trending content
        print("\nTesting trending content...")
        trending = await engine._get_trending_content(top_n=3)
        print(f"Trending content: {len(trending)} items")
        
        # Test hybrid recommendations
        print("\nTesting hybrid recommendations...")
        hybrid_recs = await engine.get_recommendations("test_user", top_n=5, strategy='hybrid')
        print(f"Hybrid recommendations: {len(hybrid_recs)} items")
        
        # Test sentiment-based filtering
        print("\nTesting sentiment-based filtering...")
        positive_content = await engine.get_content_by_sentiment(min_sentiment=0.7, limit=3)
        print(f"Positive content: {len(positive_content)} items")
    
    # Run the test
    asyncio.run(test_recommendations())
