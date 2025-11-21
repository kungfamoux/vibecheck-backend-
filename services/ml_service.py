import numpy as np
from textblob import TextBlob
from typing import List, Dict, Any, Tuple, Optional, Union
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import re
import logging
from datetime import datetime
import json
import os
import string
from heapq import nlargest
from string import punctuation
import emoji

# Download required NLTK data
REQUIRED_NLTK_DATA = [
    'punkt', 'stopwords', 'wordnet', 'omw-1.4', 
    'averaged_perceptron_tagger', 'vader_lexicon'
]

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

# Initialize NLTK components
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    """Core text analysis functionality."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(punctuation)
        self.sia = SentimentIntensityAnalyzer()
        
    def _get_wordnet_pos(self, word: str) -> str:
        """Map POS tag to first character lemmatize() accepts."""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                   "N": wordnet.NOUN,
                   "V": wordnet.VERB,
                   "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

class MLService:
    """
    Machine Learning service for natural language processing tasks including:
    - Sentiment analysis (basic and advanced)
    - Content categorization and topic modeling
    - Text summarization
    - Keyword and entity extraction
    - Mood and emotion detection
    - Text preprocessing and cleaning
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the ML service with optional pre-trained model path.
        
        Args:
            model_path: Path to a pre-trained model (not implemented in this version)
        """
        self.analyzer = TextAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize NLTK resources
        try:
            download_nltk_data()
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
            
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000
        )
        
        # Initialize topic model (LDA)
        self.lda_model = None
        self.n_topics = 10
        
        # Sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Emoji and slang support
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
            
        # Common slang and their meanings
        self.slang_mapping = {
            'lol': 'laughing out loud', 'lmao': 'laughing my ass off',
            'rofl': 'rolling on the floor laughing', 'omg': 'oh my god',
            'btw': 'by the way', 'idk': "i don't know", 'tbh': 'to be honest',
            'imo': 'in my opinion', 'smh': 'shaking my head', 'fyi': 'for your information',
            'rn': 'right now', 'ikr': 'i know right', 'tbh': 'to be honest',
            'idc': "i don't care", 'omw': 'on my way', 'nvm': 'never mind',
            'thx': 'thanks', 'pls': 'please', 'u': 'you', 'r': 'are',
            'ur': 'your', 'y': 'why', '4': 'for', '2': 'to', 'b': 'be'
        }
        
        # Content categories and mood detection
        self.content_categories = [
            'technology', 'sports', 'entertainment', 'politics', 
            'health', 'science', 'business', 'education'
        ]
        
        # Initialize sentiment analyzer (TextBlob as base, can be extended with more advanced models)
        self.sentiment_analyzer = TextBlob
        
        # Initialize mood keywords
        self.mood_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'awesome', 'delighted', 'pleased'],
            'sad': ['sad', 'unhappy', 'depressed', 'miserable', 'heartbroken', 'gloomy', 'sorrow', 'tearful', 'down', 'upset'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'outraged', 'livid', 'enraged', 'irate'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'astounded', 'dumbfounded', 'flabbergasted', 'startled'],
            'fearful': ['scared', 'afraid', 'fearful', 'terrified', 'frightened', 'panicked', 'nervous', 'worried', 'anxious', 'apprehensive'],
            'disgusted': ['disgusted', 'repulsed', 'revolted', 'sickened', 'nauseated', 'horrified', 'appalled', 'grossed out', 'displeased'],
            'neutral': ['ok', 'fine', 'alright', 'normal', 'usual', 'regular', 'typical', 'ordinary', 'commonplace', 'unremarkable']
        }
        
        # Initialize model parameters
        self.model_initialized = False
        self.model_path = model_path
        
        logger.info("MLService initialized")
    
    def preprocess_text(self, text: str, handle_emojis: bool = True, handle_slang: bool = True) -> List[str]:
        """
        Preprocess text with support for emojis and slang.
        
        Args:
            text: Input text to preprocess
            handle_emojis: Whether to process emojis
            handle_slang: Whether to expand slang terms
            
        Returns:
            List of preprocessed tokens
        """
        if not text:
            return []
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Handle emojis
            if handle_emojis and any(char in emoji.EMOJI_DATA for char in text):
                try:
                    text = emoji.demojize(text, delimiters=(' ', ' '))
                except Exception as e:
                    logger.warning(f"Error processing emojis: {e}")
                    # Remove emojis if there's an error processing them
                    text = ''.join(char for char in text if char not in emoji.EMOJI_DATA)
            
            # Handle common slang
            if handle_slang:
                words = text.split()
                expanded_words = [self.slang_mapping.get(word, word) for word in words]
                text = ' '.join(expanded_words)
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            
            # Remove special characters but keep apostrophes for contractions
            text = re.sub(r'[^\w\s\']', ' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and short tokens, and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 2 and not token.isdigit()
            ]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}", exc_info=True)
            return []
            
    def analyze_sentiment(self, text: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Advanced sentiment analysis of a given text using multiple approaches.
        
        Args:
            text: Input text to analyze
            detailed: Whether to return detailed sentiment analysis
            
        Returns:
            Dictionary with sentiment scores and analysis
        """
        if not text or not isinstance(text, str):
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0
            }
            
        try:
            # Basic TextBlob sentiment
            blob = TextBlob(text)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)
            
            # VADER sentiment analysis
            vader_scores = self.sia.polarity_scores(text)
            
            # Determine overall sentiment
            if vader_scores['compound'] >= 0.05:
                sentiment = 'positive'
                confidence = vader_scores['pos']
            elif vader_scores['compound'] <= -0.05:
                sentiment = 'negative'
                confidence = vader_scores['neg']
            else:
                sentiment = 'neutral'
                confidence = vader_scores['neu']
            
            result = {
                'polarity': polarity,  # -1 to 1
                'subjectivity': subjectivity,  # 0 to 1
                'sentiment': sentiment,
                'confidence': round(confidence, 4),
                'vader_scores': {
                    'positive': vader_scores['pos'],
                    'neutral': vader_scores['neu'],
                    'negative': vader_scores['neg'],
                    'compound': vader_scores['compound']
                }
            }
            
            # Add detailed analysis if requested
            if detailed:
                result.update({
                    'sentences': [{
                        'text': str(sent),
                        'polarity': float(sent.sentiment.polarity),
                        'subjectivity': float(sent.sentiment.subjectivity)
                    } for sent in blob.sentences]
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}", exc_info=True)
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def detect_mood(self, text: str, analyze_transitions: bool = False) -> Dict[str, Any]:
        """
        Analyze text and detect the most prominent mood with supporting details.
        Supports emojis, slang, and mood transitions in longer texts.
        
        Args:
            text: Input text to analyze
            analyze_transitions: Whether to analyze mood changes in longer texts
            
        Returns:
            Dictionary containing:
            - primary_mood: The most dominant mood detected
            - confidence: Confidence score (0-1)
            - mood_breakdown: All detected moods with their scores
            - sentiment: Overall sentiment of the text
            - mood_transitions: (if analyze_transitions=True) List of mood changes in text
        """
        try:
            if not text or not text.strip():
                return {
                    'primary_mood': 'neutral',
                    'confidence': 0.0,
                    'mood_breakdown': {},
                    'sentiment': 'neutral'
                }
            
            # Handle emojis and slang in preprocessing
            tokens = self.preprocess_text(text, handle_emojis=True, handle_slang=True)
            
            # Get sentiment analysis
            sentiment = self.analyze_sentiment(text)
            
            # Analyze mood transitions for longer texts
            mood_transitions = []
            if analyze_transitions and len(text.split()) > 20:  # Only for longer texts
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    if len(sentence.split()) > 3:  # Only analyze meaningful sentences
                        mood = self._analyze_single_mood(sentence)
                        mood_transitions.append({
                            'text': sentence,
                            'mood': mood['primary_mood'],
                            'confidence': mood['confidence']
                        })
            
            # Analyze overall mood
            mood_result = self._analyze_single_mood(text)
            
            # Add transitions if analyzed
            if analyze_transitions and mood_transitions:
                mood_result['mood_transitions'] = mood_transitions
            
            return mood_result
            
        except Exception as e:
            logger.error(f"Error in mood detection: {e}")
            return {
                'primary_mood': 'neutral',
                'confidence': 0.0,
                'mood_breakdown': {},
                'sentiment': 'neutral',
                'error': str(e)
            }
    
    def _analyze_single_mood(self, text: str) -> Dict[str, Any]:
        """Helper method to analyze mood for a single text segment."""
        sentiment = self.analyze_sentiment(text)
        tokens = self.preprocess_text(text, handle_emojis=True, handle_slang=True)
        
        mood_scores = {}
        for mood, data in self.mood_keywords.items():
            keywords = data['keywords']
            weight = data['weight']
            
            # Boost scores based on sentiment alignment
            sentiment_boost = 1.0
            if (sentiment['sentiment'] == 'positive' and mood in ['happy', 'excited', 'inspiring']) or \
               (sentiment['sentiment'] == 'negative' and mood in ['sad', 'angry', 'mysterious']):
                sentiment_boost = 1.5
                
            # Score based on keyword matches
            score = sum(
                weight * tokens.count(word)
                for word in keywords
                if word in tokens
            )
            
            # Add emoji-based mood detection
            emoji_matches = re.findall(r':[a-z_]+:', text.lower())
            for emoji_text in emoji_matches:
                if any(kw in emoji_text for kw in keywords):
                    score += 2.0  # Higher weight for emoji matches
            
            if score > 0:
                mood_scores[mood] = min(1.0, (score / 5.0) * sentiment_boost)
        
        # If no specific mood detected, use sentiment as mood
        if not mood_scores:
            primary_mood = sentiment['sentiment']
            confidence = abs(sentiment['vader_scores']['compound'])
            return {
                'primary_mood': primary_mood,
                'confidence': round(confidence, 2),
                'mood_breakdown': {primary_mood: confidence},
                'sentiment': sentiment['sentiment']
            }
        
        # Get primary mood (highest scoring)
        primary_mood, confidence = max(mood_scores.items(), key=lambda x: x[1])
        
        return {
            'primary_mood': primary_mood,
            'confidence': round(confidence, 2),
            'mood_breakdown': {k: round(v, 2) for k, v in sorted(
                mood_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )},
            'sentiment': sentiment['sentiment']
        }
    
    def extract_keywords(self, text: str, top_n: int = 10, method: str = 'tfidf') -> List[Dict[str, Any]]:
        """
        Extract top keywords from the text using different methods.
        
        Args:
            text: Input text
            top_n: Number of keywords to return
            method: Method to use ('tfidf', 'freq', 'rake')
            
        Returns:
            List of dictionaries with keywords and their scores
        """
        if not text or not isinstance(text, str):
            return []
            
        try:
            if method == 'tfidf':
                # Use TF-IDF for keyword extraction
                tfidf = self.tfidf_vectorizer.fit_transform([text])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # Get top TF-IDF scores
                sorted_items = self._sort_coo(tfidf.tocoo())
                keywords = self._extract_topn_from_vector(feature_names, sorted_items, top_n)
                return [{'word': k, 'score': float(v)} for k, v in keywords.items()]
                
            elif method == 'freq':
                # Simple frequency-based keyword extraction
                tokens = self.preprocess_text(text, remove_stopwords=True, lemmatize=True)
                word_freq = Counter(tokens)
                
                # Get most common words
                most_common = word_freq.most_common(top_n)
                return [{'word': word, 'score': float(freq)} for word, freq in most_common]
                
            else:
                raise ValueError(f"Unsupported keyword extraction method: {method}")
                
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}", exc_info=True)
            return []
    
    def _sort_coo(self, coo_matrix):
        """Sort a COO matrix by values."""
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    
    def _extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """Get the top N items from the sorted COO matrix."""
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []
        
        for idx, score in sorted_items:
            score_vals.append(round(float(score), 4))
            feature_vals.append(feature_names[idx])
        
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]
            
        return results
    
    def analyze_content(self, text: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of content including sentiment, 
        mood, topics, and key information extraction.
        
        Args:
            text: Input text to analyze
            detailed: Whether to include detailed analysis
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if not text or not isinstance(text, str):
            return {
                'error': 'Invalid input text',
                'analysis_time': datetime.utcnow().isoformat()
            }
            
        try:
            # Basic text statistics
            words = self.preprocess_text(text, remove_stopwords=False, lemmatize=False)
            sentences = sent_tokenize(text)
            
            text_stats = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': sum(len(word) for word in words) / max(1, len(words)),
                'avg_sentence_length': len(words) / max(1, len(sentences)),
                'unique_words': len(set(words)) / max(1, len(words))
            }
            
            # Sentiment analysis
            sentiment = self.analyze_sentiment(text, detailed=detailed)
            
            # Mood detection
            moods = self.detect_mood(text)
            
            # Keyword extraction (using both TF-IDF and frequency methods)
            keywords_tfidf = self.extract_keywords(text, method='tfidf')
            keywords_freq = self.extract_keywords(text, method='freq')
            
            # Named entity recognition (simplified)
            entities = self._extract_entities(text)
            
            # Topic modeling (if enough text)
            topics = []
            if len(words) > 50:  # Only run topic modeling on longer texts
                topics = self.identify_topics(text, n_topics=min(3, len(sentences)//5))
            
            # Build result dictionary
            result = {
                'text_metrics': text_stats,
                'sentiment': sentiment,
                'moods': moods,
                'keywords': {
                    'tfidf': keywords_tfidf,
                    'frequency': keywords_freq
                },
                'entities': entities,
                'topics': topics,
                'analysis_time': datetime.utcnow().isoformat()
            }
            
            # Add detailed analysis if requested
            if detailed:
                result.update({
                    'text_summary': self.summarize_text(text),
                    'readability': self.assess_readability(text),
                    'emotion_intensity': self.assess_emotion_intensity(text)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}", exc_info=True)
            return {
                'error': str(e),
                'analysis_time': datetime.utcnow().isoformat()
            }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            # This is a simplified version - consider using spaCy for production
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            chunked = nltk.ne_chunk(pos_tags, binary=False)
            
            entities = []
            current_entity = []
            
            for chunk in chunked:
                if hasattr(chunk, 'label'):
                    current_entity.append(' '.join(c[0] for c in chunk))
                    entity_type = chunk.label()
                else:
                    if current_entity:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'type': entity_type,
                            'count': 1
                        })
                        current_entity = []
            
            # Add any remaining entity
            if current_entity:
                entities.append({
                    'text': ' '.join(current_entity),
                    'type': entity_type,
                    'count': 1
                })
            
            # Merge duplicate entities
            merged_entities = {}
            for ent in entities:
                key = (ent['text'].lower(), ent['type'])
                if key in merged_entities:
                    merged_entities[key]['count'] += 1
                else:
                    merged_entities[key] = ent
            
            return list(merged_entities.values())
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def identify_topics(self, text: str, n_topics: int = 3, n_keywords: int = 5) -> List[Dict[str, Any]]:
        """
        Identify main topics in the text using LDA.
        
        Args:
            text: Input text
            n_topics: Number of topics to identify
            n_keywords: Number of keywords per topic
            
        Returns:
            List of topics with their keywords
        """
        try:
            # Preprocess text
            sentences = sent_tokenize(text)
            if len(sentences) < n_topics * 2:  # Need enough sentences for topic modeling
                return []
            
            # Create document-term matrix
            doc_term_matrix = self.count_vectorizer.fit_transform(sentences)
            
            # Train LDA model
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
            lda_model.fit(doc_term_matrix)
            
            # Extract topic keywords
            feature_names = self.count_vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_keywords = [feature_names[i] for i in topic.argsort()[:-n_keywords - 1:-1]]
                topics.append({
                    'topic_id': topic_idx,
                    'keywords': top_keywords,
                    'weight': float(topic.sum() / topic.sum())  # Normalized weight
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}", exc_info=True)
            []
    
    def summarize_text(self, text: str, ratio: float = 0.2) -> str:
        """
        Generate a summary of the text.
        
        Args:
            text: Input text
            ratio: Ratio of sentences to include in summary
            
        Returns:
            Summarized text
        """
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            # Calculate word frequencies
            word_frequencies = {}
            words = word_tokenize(text.lower())
            
            for word in words:
                if word not in self.stop_words and word not in string.punctuation:
                    if word in word_frequencies:
                        word_frequencies[word] += 1
                    else:
                        word_frequencies[word] = 1
            
            # Calculate sentence scores
            sentence_scores = {}
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word in word_frequencies:
                        if len(sentence.split(' ')) < 30:  # Avoid very long sentences
                            if sentence not in sentence_scores:
                                sentence_scores[sentence] = word_frequencies[word]
                            else:
                                sentence_scores[sentence] += word_frequencies[word]
            
            # Select top sentences
            summary_sentences = nlargest(
                max(1, int(len(sentences) * ratio)),
                sentence_scores,
                key=sentence_scores.get
            )
            
            # Join sentences to form summary
            summary = ' '.join(summary_sentences)
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:500] + '...'  # Return first 500 chars as fallback
    
    def assess_readability(self, text: str) -> Dict[str, float]:
        """
        Assess the readability of the text using various metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with readability scores
        """
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Average sentence length
            avg_sentence_length = len(words) / max(1, len(sentences))
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / max(1, len(words))
            
            # Flesch Reading Ease
            # Higher score = easier to read (90-100: Very Easy, 60-70: Standard, 0-30: Very Difficult)
            total_syllables = sum([self._count_syllables(word) for word in words])
            flesch_score = 206.835 - (1.015 * (len(words) / max(1, len(sentences)))) - (84.6 * (total_syllables / max(1, len(words))))
            
            return {
                'flesch_reading_ease': round(flesch_score, 2),
                'avg_sentence_length': round(avg_sentence_length, 2),
                'avg_word_length': round(avg_word_length, 2),
                'word_count': len(words),
                'sentence_count': len(sentences)
            }
            
        except Exception as e:
            logger.error(f"Error assessing readability: {e}")
            return {}
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable counting."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        
        if word[0] in vowels:
            count += 1
            
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
                
        if word.endswith('e'):
            count -= 1
            
        return max(1, count)
    
    def assess_emotion_intensity(self, text: str) -> Dict[str, float]:
        """
        Assess the emotional intensity of the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotion intensity scores
        """
        try:
            # Use VADER for emotion intensity
            sentiment = self.sia.polarity_scores(text)
            
            # Calculate overall emotional intensity
            intensity = (abs(sentiment['pos'] - sentiment['neg']) + sentiment['compound']) / 2
            
            return {
                'positive': sentiment['pos'],
                'negative': sentiment['neg'],
                'neutral': sentiment['neu'],
                'compound': sentiment['compound'],
                'intensity': abs(intensity)
            }
            
        except Exception as e:
            logger.error(f"Error assessing emotion intensity: {e}")
            return {}
    
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
