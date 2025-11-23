import os
import logging
import asyncio
from typing import List, Dict, Optional, Any
import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class RateLimiter:
    """Simple rate limiter for API requests"""
    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.timestamps = []
        
    async def wait(self):
        now = asyncio.get_event_loop().time()
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]
        
        if len(self.timestamps) >= self.max_requests:
            sleep_time = self.period - (now - self.timestamps[0]) + 0.1
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.timestamps.append(asyncio.get_event_loop().time())

class YouTubeService:
    BASE_URL = "https://www.googleapis.com/youtube/v3"
    
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        self.rate_limiter = RateLimiter(max_requests=90, period=60)  # 90 requests per minute
        
        if not self.api_key:
            logger.warning("YouTube API key not found. YouTube features will be disabled.")
    
    async def search_videos(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for YouTube videos matching the query."""
        if not self.api_key:
            return []
            
        await self.rate_limiter.wait()
        
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': min(max_results, 50),
            'key': self.api_key,
            'videoEmbeddable': 'true',
            'videoDuration': 'medium'
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.BASE_URL}/search",
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()
                
                items = response.json().get('items', [])
                return [{
                    'id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'thumbnail': item['snippet']['thumbnails'].get('high', {}).get('url', ''),
                    'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    'source': 'youtube',
                    'type': 'video',
                    'channel': item['snippet'].get('channelTitle', ''),
                    'published_at': item['snippet'].get('publishedAt', '')
                } for item in items]
                
            except Exception as e:
                logger.error(f"Error searching YouTube videos: {e}")
                return []

class SpotifyService:
    BASE_URL = "https://api.spotify.com/v1"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.access_token = None
        self.token_expiry = 0
        self.rate_limiter = RateLimiter(max_requests=50, period=10)  # 50 requests per 10 seconds
        
        if not all([self.client_id, self.client_secret]):
            logger.warning("Spotify credentials not found. Spotify features will be disabled.")
    
    async def _get_access_token(self) -> bool:
        """Get an access token using client credentials."""
        if not all([self.client_id, self.client_secret]):
            return False
            
        auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        headers = {
            'Authorization': f'Basic {auth}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {'grant_type': 'client_credentials'}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.TOKEN_URL, headers=headers, data=data)
                response.raise_for_status()
                
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.token_expiry = asyncio.get_event_loop().time() + token_data.get('expires_in', 3600) - 300  # 5 min buffer
                return True
                
        except Exception as e:
            logger.error(f"Error getting Spotify access token: {e}")
            return False
    
    async def _ensure_token(self) -> bool:
        """Ensure we have a valid access token."""
        if not self.access_token or asyncio.get_event_loop().time() >= self.token_expiry:
            return await self._get_access_token()
        return True
    
    async def search_playlists(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for Spotify playlists."""
        if not await self._ensure_token():
            return []
            
        await self.rate_limiter.wait()
        
        params = {
            'q': query,
            'type': 'playlist',
            'limit': min(limit, 50)
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/search",
                    params=params,
                    headers={'Authorization': f'Bearer {self.access_token}'},
                    timeout=10.0
                )
                response.raise_for_status()
                
                playlists = response.json().get('playlists', {}).get('items', [])
                return [{
                    'id': playlist['id'],
                    'name': playlist['name'],
                    'description': playlist.get('description', ''),
                    'url': playlist['external_urls']['spotify'],
                    'thumbnail': playlist['images'][0]['url'] if playlist.get('images') else None,
                    'tracks_count': playlist.get('tracks', {}).get('total', 0),
                    'source': 'spotify',
                    'type': 'playlist'
                } for playlist in playlists]
                
        except Exception as e:
            logger.error(f"Error searching Spotify playlists: {e}")
            return []
    
    async def get_recommendations(
        self,
        seed_genres: List[str] = None,
        limit: int = 10,
        **audio_features
    ) -> List[Dict[str, Any]]:
        """Get track recommendations based on audio features."""
        if not await self._ensure_token():
            return []
            
        await self.rate_limiter.wait()
        
        params = {
            'limit': min(limit, 100),
            'market': 'US'  # Required by Spotify
        }
        
        if seed_genres:
            params['seed_genres'] = ','.join(seed_genres[:5])
            
        # Add audio features if provided
        for key, value in audio_features.items():
            if value is not None:
                params[f'target_{key}'] = value
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}/recommendations",
                    params=params,
                    headers={'Authorization': f'Bearer {self.access_token}'},
                    timeout=10.0
                )
                response.raise_for_status()
                
                tracks = response.json().get('tracks', [])
                return [{
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'duration_ms': track['duration_ms'],
                    'thumbnail': track['album']['images'][0]['url'] if track['album'].get('images') else None,
                    'source': 'spotify',
                    'type': 'track'
                } for track in tracks]
                
        except Exception as e:
            logger.error(f"Error getting Spotify recommendations: {e}")
            return []

# Create global instances
youtube_service = YouTubeService()
spotify_service = SpotifyService()
