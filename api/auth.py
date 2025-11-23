from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from typing import Optional
import httpx
import os
import base64
import json
import logging
from datetime import datetime, timedelta

from ..models.schemas import Token, UserResponse
from ..services.content_services import spotify_service

router = APIRouter(tags=["Authentication"])
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.spotify.com/authorize",
    tokenUrl="https://accounts.spotify.com/api/token"
)

# Environment variables
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# Token storage (in production, use a database)
user_tokens = {}

def get_spotify_auth_url() -> str:
    """Generate Spotify OAuth2 authorization URL"""
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "user-read-private user-read-email user-read-playback-state user-modify-playback-state",
        "show_dialog": "true"
    }
    query_params = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"https://accounts.spotify.com/authorize?{query_params}"

@router.get("/login/spotify")
async def login_spotify():
    """Initiate Spotify OAuth flow"""
    return {"auth_url": get_spotify_auth_url()}

@router.get("/auth/spotify/callback")
async def spotify_callback(code: str, state: Optional[str] = None):
    """Handle Spotify OAuth callback"""
    try:
        # Exchange authorization code for access token
        token_url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            response.raise_for_status()
            token_data = response.json()
            
        # Get user info
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        async with httpx.AsyncClient() as client:
            user_response = await client.get("https://api.spotify.com/v1/me", headers=headers)
            user_response.raise_for_status()
            user_data = user_response.json()
        
        # Store tokens (in production, use a database)
        user_id = user_data["id"]
        user_tokens[user_id] = {
            "access_token": token_data["access_token"],
            "refresh_token": token_data["refresh_token"],
            "expires_at": datetime.utcnow() + timedelta(seconds=token_data["expires_in"]),
            "user_info": user_data
        }
        
        # Return tokens and user info (in production, issue a JWT instead)
        return {
            "access_token": token_data["access_token"],
            "refresh_token": token_data["refresh_token"],
            "token_type": "bearer",
            "expires_in": token_data["expires_in"],
            "user": user_data
        }
        
    except httpx.HTTPStatusError as e:
        logging.error(f"Spotify API error: {e.response.text}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to authenticate with Spotify"
        )
    except Exception as e:
        logging.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during authentication"
        )

@router.get("/auth/me", response_model=UserResponse)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user info using the access token"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.spotify.com/v1/me", headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
