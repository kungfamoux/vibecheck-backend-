from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from .ml_service import ml_service
from models.database import content_ref

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Handles content analysis and feature extraction."""
    
    def __init__(self):
        # Initialize vectorizers for different content aspects
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        self.count_vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Topic model
        self.lda = None
        self.n_topics = 10
        
    def extract_content_features(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from content for recommendation."""
        try:
            # Combine text fields for analysis
            text = f"{content.get('title', '')} {content.get('description', '')} {' '.join(content.get('tags', []))}"
            
            # Get ML-based analysis
            analysis = ml_service.analyze_content(text)
            
            # Extract key features
            features = {
                'sentiment': analysis.get('sentiment', {}).get('sentiment', 'neutral'),
                'sentiment_score': analysis.get('sentiment', {}).get('compound', 0),
                'moods': [mood['mood'] for mood in analysis.get('moods', [])[:3]],
                'top_keywords': [kw['word'] for kw in analysis.get('keywords', {}).get('tfidf', [])[:5]],
                'entities': [e['text'] for e in analysis.get('entities', [])[:5]],
                'readability': analysis.get('readability', {}).get('flesch_reading_ease', 60),
                'topic_distribution': self._get_topic_distribution(text)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting content features: {e}")
            return {}
    
    def _get_topic_distribution(self, text: str) -> List[float]:
        """Get topic distribution for the given text."""
        if not self.lda:
            return []
            
        try:
            # Transform text to document-term matrix
            dtm = self.count_vectorizer.transform([text])
            
            # Get topic distribution
            topic_dist = self.lda.transform(dtm)
            return topic_dist[0].tolist()
            
        except Exception as e:
            logger.error(f"Error in topic distribution: {e}")
            return []


class UserProfileManager:
    """Manages user profiles and preferences."""
    
    def __init__(self):
        self.user_profiles = {}
        
    def update_user_profile(self, user_id: str, interaction: Dict[str, Any]):
        """Update user profile based on interaction."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interaction_history': [],
                'preferences': {
                    'topics': {},
                    'moods': {},
                    'sentiment': 0.0,
                    'content_types': {}
                },
                'last_updated': datetime.utcnow()
            }
        
        # Add interaction to history
        self.user_profiles[user_id]['interaction_history'].append(interaction)
        
        # Update preferences based on interaction
        self._update_preferences(user_id, interaction)
        
    def _update_preferences(self, user_id: str, interaction: Dict[str, Any]):
        """Update user preferences based on interaction."""
        content_id = interaction.get('content_id')
        content = self._get_content(content_id)
        
        if not content:
            return
            
        # Get content features
        features = self._extract_content_features(content)
        
        # Update topic preferences
        for topic, score in features.get('topic_distribution', {}).items():
            self.user_profiles[user_id]['preferences']['topics'][topic] = \
                self.user_profiles[user_id]['preferences']['topics'].get(topic, 0) + score
        
        # Update mood preferences
        for mood in features.get('moods', []):
            self.user_profiles[user_id]['preferences']['moods'][mood] = \
                self.user_profiles[user_id]['preferences']['moods'].get(mood, 0) + 1
        
        # Update sentiment preference (running average)
        current_sentiment = self.user_profiles[user_id]['preferences']['sentiment']
        interaction_count = len(self.user_profiles[user_id]['interaction_history'])
        new_sentiment = (current_sentiment * (interaction_count - 1) + features.get('sentiment_score', 0)) / interaction_count
        self.user_profiles[user_id]['preferences']['sentiment'] = new_sentiment
        
        # Update content type preferences
        content_type = content.get('type', 'unknown')
        self.user_profiles[user_id]['preferences']['content_types'][content_type] = \
            self.user_profiles[user_id]['preferences']['content_types'].get(content_type, 0) + 1
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences with normalized scores."""
        if user_id not in self.user_profiles:
            return {}
            
        preferences = self.user_profiles[user_id]['preferences']
        
        # Normalize scores
        def normalize_scores(scores):
            total = sum(scores.values())
            return {k: v/total if total > 0 else 0 for k, v in scores.items()}
        
        return {
            'topics': normalize_scores(preferences['topics']),
            'moods': normalize_scores(preferences['moods']),
            'sentiment': preferences['sentiment'],
            'content_types': normalize_scores(preferences['content_types'])
        }
    
    def _get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content by ID (placeholder - implement actual content retrieval)."""
        # This should be replaced with actual content retrieval from your database
        return {}
    
    def _extract_content_features(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from content for preference analysis."""
        # This should be implemented based on your content structure
        return {}


class MoodBasedRecommender:
    """
    A simple recommendation engine that suggests content based on mood.
    Uses text analysis to find content that matches the user's current mood.
    """
    
    def __init__(self):
        self.content_cache = {}
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.content_embeddings = None
        logger.info("MoodBasedRecommender initialized")
    
    async def _refresh_content_cache(self):
        """Load and process content from the database."""
        try:
            content_docs = content_ref.stream()
            self.content_cache = {
                doc.id: {**doc.to_dict(), 'content_id': doc.id}
                for doc in content_docs
            }
            
            # Prepare text for vectorization
            content_texts = [
                f"{content.get('title', '')} {content.get('description', '')} "
                f"{' '.join(content.get('tags', []))}"
                for content in self.content_cache.values()
            ]
            
            # Create TF-IDF vectors
            if content_texts:
                self.content_embeddings = self.vectorizer.fit_transform(content_texts)
            
            logger.info(f"Loaded {len(self.content_cache)} content items")
            
        except Exception as e:
            logger.error(f"Error refreshing content cache: {e}")
            self.content_cache = {}
            self.content_embeddings = None
    
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
    
    async def get_similar_mood_content(
        self, 
        mood: str,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find content that matches a specific mood.
        
        Args:
            mood: The target mood to match (e.g., 'happy', 'calm', 'energetic')
            top_n: Number of items to return
            
        Returns:
            List of content items with mood match scores
        """
        try:
            if not self.content_cache:
                await self._refresh_content_cache()
                
            if not self.content_cache:
                return []
            
            # Analyze mood for all content
            mood_scores = []
            
            for content_id, content in self.content_cache.items():
                # Get text content
                text = f"{content.get('title', '')} {content.get('description', '')}"
                
                # Analyze mood
                analysis = ml_service.analyze_content(text)
                content_mood = next(
                    (m['mood'] for m in analysis.get('moods', []) if m['mood'].lower() == mood.lower()),
                    None
                )
                
                if content_mood:
                    mood_scores.append((content_id, 1.0))
            
            # Sort by mood match score
            mood_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N matches
            results = []
            for content_id, score in mood_scores[:top_n]:
                results.append({
                    **self.content_cache[content_id],
                    'content_id': content_id,
                    'mood_match_score': score,
                    'explanation': f"Matches mood: {mood}"
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error finding mood-matching content: {e}")
            return []
    
    def _calculate_content_similarity(self, content_id1: str, content_id2: str) -> float:
        """Calculate content similarity using TF-IDF and other features."""
        if content_id1 not in self.content_cache or content_id2 not in self.content_cache:
            return 0.0
            
        # Get content indices
        content_ids = list(self.content_cache.keys())
        try:
            idx1 = content_ids.index(content_id1)
            idx2 = content_ids.index(content_id2)
            
            # Use precomputed similarity matrix
            if self.similarity_matrix is not None and \
               idx1 < self.similarity_matrix.shape[0] and \
               idx2 < self.similarity_matrix.shape[1]:
                return float(self.similarity_matrix[idx1][idx2])
                
        except (ValueError, IndexError):
            pass
            
        # Fallback to simple text similarity
        text1 = self._get_content_text(content_id1)
        text2 = self._get_content_text(content_id2)
        
        if not text1 or not text2:
            return 0.0
            
        # Use ML service for more accurate text similarity
        return ml_service.calculate_similarity(text1, text2)
    
    def _calculate_topic_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate topic distribution similarity."""
        topics1 = features1.get('topic_distribution', [])
        topics2 = features2.get('topic_distribution', [])
        
        if not topics1 or not topics2:
            return 0.0
            
        # Calculate cosine similarity between topic distributions
        return float(cosine_similarity([topics1], [topics2])[0][0])
    
    def _calculate_sentiment_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate sentiment similarity."""
        sent1 = features1.get('sentiment_score', 0)
        sent2 = features2.get('sentiment_score', 0)
        
        # Calculate similarity on a scale from 0 to 1
        return 1.0 - (abs(sent1 - sent2) / 2.0)  # Normalize to 0-1 range
    
    def _calculate_mood_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate mood similarity using Jaccard index."""
        moods1 = set(features1.get('moods', []))
        moods2 = set(features2.get('moods', []))
        
        if not moods1 or not moods2:
            return 0.0
            
        intersection = len(moods1.intersection(moods2))
        union = len(moods1.union(moods2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_content_text(self, content_id: str) -> str:
        """Get text content for a given content ID."""
        if content_id not in self.content_cache:
            return ""
            
        content = self.content_cache[content_id]
        return f"{content.get('title', '')} {content.get('description', '')} {' '.join(content.get('tags', []))}"
    
    def _calculate_content_weights(self, user_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weights for user's historical content interactions."""
        content_weights = {}
        max_interactions = len(user_history)
        
        # Apply time decay to older interactions
        for i, interaction in enumerate(user_history):
            content_id = interaction.get('content_id')
            if not content_id:
                continue
                
            # Recent interactions get higher weights
            recency_weight = (max_interactions - i) / max_interactions
            
            # Interaction type weights (e.g., like > view)
            interaction_type = interaction.get('type', 'view')
            type_weight = {
                'like': 1.5,
                'save': 1.3,
                'share': 1.4,
                'comment': 1.2,
                'view': 1.0
            }.get(interaction_type, 1.0)
            
            # Combine weights
            content_weights[content_id] = recency_weight * type_weight
        
        # Normalize weights
        max_weight = max(content_weights.values()) if content_weights else 1.0
        if max_weight > 0:
            content_weights = {k: v/max_weight for k, v in content_weights.items()}
            
        return content_weights
    
    def _calculate_preference_match(self, content_id: str, user_prefs: Dict[str, Any]) -> float:
        """Calculate how well content matches user preferences."""
        if not user_prefs or content_id not in self.content_features:
            return 0.0
            
        content_features = self.content_features[content_id]
        score = 0.0
        
        # Topic match
        content_topics = set(content_features.get('topics', []))
        user_topics = set(user_prefs.get('topics', {}).keys())
        topic_match = len(content_topics.intersection(user_topics)) / len(user_topics) if user_topics else 0.0
        
        # Mood match
        content_moods = set(content_features.get('moods', []))
        user_moods = set(user_prefs.get('moods', {}).keys())
        mood_match = len(content_moods.intersection(user_moods)) / len(user_moods) if user_moods else 0.0
        
        # Sentiment match
        content_sentiment = content_features.get('sentiment_score', 0)
        user_sentiment = user_prefs.get('sentiment', 0)
        sentiment_match = 1.0 - abs(content_sentiment - user_sentiment) / 2.0  # Normalize to 0-1
        
        # Content type match
        content_type = self.content_cache.get(content_id, {}).get('type', 'unknown')
        type_match = user_prefs.get('content_types', {}).get(content_type, 0.0)
        
        # Weighted combination of all factors
        weights = {
            'topic': 0.4,
            'mood': 0.2,
            'sentiment': 0.3,
            'type': 0.1
        }
        
        score = (
            weights['topic'] * topic_match +
            weights['mood'] * mood_match +
            weights['sentiment'] * sentiment_match +
            weights['type'] * type_match
        )
        
        return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
    
    def _calculate_freshness_score(self, content: Dict[str, Any]) -> float:
        """Calculate freshness score based on content age."""
        if 'created_at' not in content:
            return 0.5  # Neutral score if no creation date
            
        try:
            # Parse creation date (assuming ISO format)
            created_at = datetime.fromisoformat(content['created_at'].replace('Z', '+00:00'))
            age_days = (datetime.utcnow() - created_at).days
            
            # Apply exponential decay (halflife of 30 days)
            halflife = 30
            return 0.5 ** (age_days / halflife)
            
        except (ValueError, TypeError):
            return 0.5
    
    def _is_diverse_enough(
        self, 
        candidate: Dict[str, Any], 
        selected: List[Dict[str, Any]], 
        min_diversity: float
    ) -> bool:
        """Check if candidate is diverse enough from already selected items."""
        if not selected:
            return True
            
        # Calculate max similarity to any selected item
        max_similarity = 0.0
        candidate_id = candidate['content_id']
        
        for item in selected:
            item_id = item['content_id']
            similarity = self._calculate_content_similarity(candidate_id, item_id)
            max_similarity = max(max_similarity, similarity)
            
            # Early exit if already too similar
            if max_similarity > (1.0 - min_diversity):
                return False
                
        return True
    
    def _generate_recommendation_explanation(
        self, 
        item: Dict[str, Any], 
        user_prefs: Dict[str, Any]
    ) -> str:
        """Generate a human-readable explanation for the recommendation."""
        content_id = item['content_id']
        content_features = self.content_features.get(content_id, {})
        
        # Get top matching aspects
        matching_aspects = []
        
        # Topic match
        content_topics = set(content_features.get('topics', []))
        user_topics = set(user_prefs.get('topics', {}).keys())
        common_topics = content_topics.intersection(user_topics)
        if common_topics:
            matching_aspects.append(f"topics like {', '.join(common_topics)}")
        
        # Mood match
        content_moods = set(content_features.get('moods', []))
        user_moods = set(user_prefs.get('moods', {}).keys())
        common_moods = content_moods.intersection(user_moods)
        if common_moods:
            matching_aspects.append(f"{', '.join(common_moods)} mood")
        
        # Content type
        content_type = self.content_cache.get(content_id, {}).get('type')
        if content_type and user_prefs.get('content_types', {}).get(content_type, 0) > 0.5:
            matching_aspects.append(f"type '{content_type}'")
        
        # Generate explanation
        if matching_aspects:
            return f"Recommended because you like {', '.join(matching_aspects)}"
        elif item['freshness_score'] > 0.7:
            return "New content you might find interesting"
        else:
            return "Popular content you might enjoy"
    
    def _get_fallback_recommendations(self, top_n: int) -> List[Dict[str, Any]]:
        """Get fallback recommendations when user has no history."""
        # Sort by popularity (or any other fallback metric)
        sorted_content = sorted(
            self.content_cache.values(),
            key=lambda x: x.get('popularity', 0),
            reverse=True
        )
        
        return [
            {
                **content,
                'content_id': content_id,
                'recommendation_score': 0.5,  # Neutral score
                'explanation': 'Popular content you might enjoy'
            }
            for content in sorted_content[:top_n]
        ]
    
    async def get_collaborative_recommendations(
        self, 
        user_id: str, 
        top_n: int = 5,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Get collaborative filtering recommendations using user-user similarity.
        
        Args:
            user_id: ID of the target user
            top_n: Number of recommendations to return
            min_similarity: Minimum user similarity threshold
            
        Returns:
            List of recommended content items with scores
        """
        try:
            # Ensure user similarity is up to date
            await self._update_user_similarities()
            
            # Get similar users
            similar_users = [
                (other_user_id, sim) 
                for other_user_id, sim in self.user_similarity.get(user_id, {}).items()
                if sim >= min_similarity and other_user_id != user_id
            ]
            
            if not similar_users:
                logger.info(f"No similar users found for {user_id}")
                return []
            
            # Get content liked by similar users
            content_scores = defaultdict(float)
            content_interactions = defaultdict(int)
            
            for other_user_id, similarity in similar_users:
                # Get other user's liked content
                user_interactions = await self._get_user_interactions(other_user_id)
                
                for interaction in user_interactions:
                    if interaction.get('type') in ['like', 'save', 'share']:
                        content_id = interaction['content_id']
                        content_scores[content_id] += similarity
                        content_interactions[content_id] += 1
            
            # Get content not already seen by the user
            user_interactions = await self._get_user_interactions(user_id)
            seen_content = {i['content_id'] for i in user_interactions}
            
            # Score and sort content
            scored_content = []
            for content_id, score in content_scores.items():
                if content_id in seen_content or content_id not in self.content_cache:
                    continue
                    
                # Normalize by number of interactions
                interaction_count = content_interactions[content_id]
                normalized_score = score / interaction_count if interaction_count > 0 else 0
                
                scored_content.append({
                    'content_id': content_id,
                    'score': normalized_score,
                    'interaction_count': interaction_count
                })
            
            # Sort by score and get top N
            scored_content.sort(key=lambda x: x['score'], reverse=True)
            
            # Prepare results
            recommendations = []
            for item in scored_content[:top_n]:
                content = self.content_cache.get(item['content_id'], {})
                recommendations.append({
                    **content,
                    'content_id': item['content_id'],
                    'recommendation_score': item['score'],
                    'explanation': f"Recommended by {item['interaction_count']} similar users"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}", exc_info=True)
            return []
    
    async def _update_user_similarities(self):
        """Update user similarity matrix based on interaction history."""
        try:
            # Get all users
            users = set()
            interactions_by_user = {}
            
            # This would be more efficient with batch queries in production
            async for doc in user_profiles_ref.stream():
                user_id = doc.id
                users.add(user_id)
                interactions = await self._get_user_interactions(user_id)
                interactions_by_user[user_id] = {
                    'interactions': interactions,
                    'content_ids': {i['content_id'] for i in interactions}
                }
            
            # Calculate Jaccard similarity between all user pairs
            user_ids = list(users)
            n_users = len(user_ids)
            
            for i in range(n_users):
                user1 = user_ids[i]
                self.user_similarity.setdefault(user1, {})
                
                for j in range(i + 1, n_users):
                    user2 = user_ids[j]
                    
                    # Calculate Jaccard similarity
                    content1 = interactions_by_user[user1]['content_ids']
                    content2 = interactions_by_user[user2]['content_ids']
                    
                    if not content1 or not content2:
                        similarity = 0.0
                    else:
                        intersection = len(content1.intersection(content2))
                        union = len(content1.union(content2))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    # Store similarity in both directions
                    self.user_similarity[user1][user2] = similarity
                    self.user_similarity.setdefault(user2, {})[user1] = similarity
            
            logger.info(f"Updated user similarity matrix for {n_users} users")
            
        except Exception as e:
            logger.error(f"Error updating user similarities: {e}", exc_info=True)
    
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
        Get personalized recommendations based on user's preferences and history.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            interaction_types: Optional list of interaction types to filter by
            
        Returns:
            List of recommended content items with scores
        """
        try:
            # Get user's interaction history
            user_history = await self._get_user_interactions(user_id, interaction_types=interaction_types)
            
            if not user_history:
                return []
            
            # Get content scores based on user preferences and history
            content_scores = {}
            
            for content_id, content in self.content_cache.items():
                # Skip content already seen by the user
                if any(i.get('content_id') == content_id for i in user_history):
                    continue
                
                # Calculate score based on various factors
                score = 0.0
                
                # Add base score based on content popularity
                score += content.get('popularity', 0) * 0.2
                
                # Add score based on content freshness
                if 'created_at' in content:
                    try:
                        created_at = datetime.fromisoformat(content['created_at'].replace('Z', '+00:00'))
                        age_days = (datetime.utcnow() - created_at).days
                        freshness = max(0, 1 - (age_days / 30))  # 30-day half-life
                        score += freshness * 0.3
                    except (ValueError, TypeError):
                        pass
                
                # Add score based on content type preference
                content_type = content.get('type')
                if content_type and content_type in self.user_preferences.get('content_types', {}):
                    score += self.user_preferences['content_types'][content_type] * 0.5
                
                content_scores[content_id] = score
            
            # Sort by score and get top N
            sorted_items = sorted(
                content_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            return [
                {
                    **self.content_cache[content_id],
                    'recommendation_score': score
                }
                for content_id, score in sorted_items
            ]
            
        except Exception as e:
            logger.error(f"Error in personalized recommendations: {e}", exc_info=True)
            return []
    
    async def _get_user_interactions(
        self,
        user_id: str,
        limit: Optional[int] = None,
        interaction_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a user's interaction history with filtering options.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of interactions to return
            interaction_types: Filter by specific interaction types
            
        Returns:
            List of user interactions with additional metadata
        """
        try:
            # Build the query
            query = interactions_ref.where('user_id', '==', user_id)
            
            # Apply interaction type filter if provided
            if interaction_types:
                query = query.where('type', 'in', interaction_types)
            
            # Apply limit if provided
            if limit:
                query = query.limit(limit)
            
            # Execute query and process results
            interactions = []
            async for doc in query.stream():
                interaction = doc.to_dict()
                interaction['interaction_id'] = doc.id
                
                # Add content details if available in cache
                content_id = interaction.get('content_id')
                if content_id and content_id in self.content_cache:
                    interaction['content'] = self.content_cache[content_id]
                
                interactions.append(interaction)
            
            # Sort by timestamp (newest first)
            interactions.sort(
                key=lambda x: x.get('timestamp', '1970-01-01'), 
                reverse=True
            )
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting user interactions: {e}", exc_info=True)
            return []
    
    async def log_interaction(self, interaction: Dict[str, Any]) -> bool:
        """
        Log a user interaction with content.
        
        Args:
            interaction: Dictionary containing interaction details
                - user_id: ID of the user
                - content_id: ID of the content
                - type: Type of interaction (view, like, save, share, etc.)
                - metadata: Additional metadata about the interaction
                
        Returns:
            bool: True if interaction was logged successfully
        """
        try:
            # Validate required fields
            required_fields = ['user_id', 'content_id', 'type']
            for field in required_fields:
                if field not in interaction:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Add timestamp if not provided
            if 'timestamp' not in interaction:
                interaction['timestamp'] = datetime.utcnow().isoformat()
            
            # Save to database
            doc_ref = await interactions_ref.add(interaction)
            
            # Update user profile with this interaction
            self.user_profile_manager.update_user_profile(
                interaction['user_id'],
                interaction
            )
            
            logger.info(f"Logged interaction: {interaction['type']} for user {interaction['user_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}", exc_info=True)
            return False
    
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
