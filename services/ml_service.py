import numpy as np
from textblob import TextBlob
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from datetime import datetime
import json
import os

# Download required NLTK data
REQUIRED_NLTK_DATA = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']

def download_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        for data in REQUIRED_NLTK_DATA:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

download_nltk_data()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLService:
    """
    Machine Learning service for natural language processing tasks including:
    - Sentiment analysis
    - Content tagging
    - Text preprocessing
    - Keyword extraction
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the ML service with optional pre-trained model path.
        
        Args:
            model_path: Path to a pre-trained model (not implemented in this version)
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize TF-IDF vectorizer for text processing
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Mood and content categories with associated keywords
        self.mood_categories = [
            'happy', 'sad', 'angry', 'excited', 'relaxed', 
            'adventurous', 'romantic', 'mysterious', 'thrilling', 'inspiring'
        ]
        
        # Enhanced keyword mapping with weights
        self.mood_keywords = {
            'happy': {'keywords': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing'], 'weight': 1.0},
            'sad': {'keywords': ['sad', 'unhappy', 'depressed', 'miserable', 'heartbroken'], 'weight': 1.0},
            'angry': {'keywords': ['angry', 'mad', 'furious', 'annoyed', 'irritated'], 'weight': 1.0},
            'excited': {'keywords': ['excited', 'thrilled', 'pumped', 'eager', 'enthusiastic'], 'weight': 1.0},
            'relaxed': {'keywords': ['relaxed', 'calm', 'peaceful', 'serene', 'chill'], 'weight': 0.9},
            'adventurous': {'keywords': ['adventure', 'explore', 'thrill', 'daring', 'bold'], 'weight': 0.9},
            'romantic': {'keywords': ['romantic', 'love', 'passion', 'affection', 'intimate'], 'weight': 0.9},
            'mysterious': {'keywords': ['mystery', 'suspense', 'enigma', 'puzzle', 'secret'], 'weight': 0.8},
            'thrilling': {'keywords': ['thrill', 'excitement', 'adrenaline', 'intense', 'gripping'], 'weight': 0.8},
            'inspiring': {'keywords': ['inspire', 'motivate', 'empower', 'encourage', 'uplift'], 'weight': 1.0}
        }
        
        # Initialize sentiment analyzer (TextBlob as base, can be extended with more advanced models)
        self.sentiment_analyzer = TextBlob
        
        # Initialize model parameters
        self.model_initialized = False
        self.model_path = model_path
        
        logger.info("MLService initialized")
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Preprocess text for analysis.
        
        Args:
            text: Input text to preprocess
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of preprocessed tokens
        """
        if not text or not isinstance(text, str):
            return []
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords if needed
            if remove_stopwords:
                tokens = [word for word in tokens if word not in self.stop_words]
                
            # Lemmatization
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return []
            
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores (polarity, subjectivity)
        """
        try:
            if not text or not isinstance(text, str):
                return {'polarity': 0.0, 'subjectivity': 0.0}
                
            # Create TextBlob object
            blob = self.sentiment_analyzer(text)
            
            return {
                'polarity': float(blob.sentiment.polarity),  # -1 to 1
                'subjectivity': float(blob.sentiment.subjectivity)  # 0 to 1
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def detect_mood(self, text: str, top_n: int = 3) -> List[Dict[str, float]]:
        """
        Detect the mood of a given text based on keyword matching.
        
        Args:
            text: Input text to analyze
            top_n: Number of top moods to return
            
        Returns:
            List of dictionaries with mood and confidence score
        """
        try:
            if not text or not isinstance(text, str):
                return []
                
            # Preprocess text
            tokens = self.preprocess_text(text, remove_stopwords=True)
            text_str = ' '.join(tokens)
            
            # Calculate scores for each mood category
            mood_scores = []
            
            for mood, data in self.mood_keywords.items():
                keywords = data['keywords']
                weight = data['weight']
                
                # Count keyword matches
                matches = sum(1 for keyword in keywords if keyword in text_str)
                
                # Calculate score (normalized by number of keywords)
                score = (matches / len(keywords)) * weight if keywords else 0
                
                if score > 0:
                    mood_scores.append({
                        'mood': mood,
                        'score': round(score, 4)
                    })
            
            # Sort by score in descending order
            mood_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top N moods
            return mood_scores[:top_n]
            
        except Exception as e:
            logger.error(f"Error in mood detection: {e}")
            return []
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract top keywords from the text.
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            
        Returns:
            List of top keywords
        """
        try:
            if not text or not isinstance(text, str):
                return []
                
            # Preprocess text
            tokens = self.preprocess_text(text, remove_stopwords=True)
            
            # Calculate word frequencies
            word_freq = {}
            for word in tokens:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
            
            # Sort by frequency and get top N
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, _ in sorted_words[:top_n]]
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            return []
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        sentiment = self.analyze_sentiment(text)
        moods = self.detect_mood(text)
        keywords = self.extract_keywords(text)
        
        return {
            'sentiment': sentiment,
            'moods': moods,
            'keywords': keywords,
            'analysis_time': datetime.utcnow().isoformat()
        }
    
    def predict_content_tags(self, text: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Predict content tags based on the input text.
        Returns a list of (tag, confidence_score) tuples.
        """
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Simple keyword matching approach (replace with ML model in production)
            tag_scores = {}
            
            for tag, keywords in self.mood_keywords.items():
                score = sum(1 for keyword in keywords if keyword in processed_text)
                if score > 0:
                    # Normalize score to 0-1 range
                    normalized_score = min(score / len(keywords), 1.0)
                    tag_scores[tag] = normalized_score
            
            # Sort by score and get top N
            sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_tags[:top_n]
            
        except Exception as e:
            logger.error(f"Error in content tag prediction: {e}")
            return []
    
    def analyze_content_feedback(self, comment: str) -> Dict[str, Any]:
        """
        Analyze user feedback on content.
        Returns a dictionary with sentiment analysis results.
        """
        sentiment_score = self.analyze_sentiment(comment)
        
        # Categorize sentiment
        if sentiment_score > 0.2:
            sentiment_label = "positive"
        elif sentiment_score < -0.2:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "analyzed_text": comment
        }

# Create a global instance of the ML service
ml_service = MLService()
