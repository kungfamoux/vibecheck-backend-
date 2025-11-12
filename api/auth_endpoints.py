from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict, Any
import logging

from models import schemas
from auth.auth_utils import auth_service
from auth.auth_bearer import JWTBearer

auth_scheme = JWTBearer()

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Authentication"])

class RegisterRequest(schemas.UserCreate):
    """Request model for user registration"""
    password: str

@router.post("/auth/register", response_model=Dict[str, Any])
async def register_user(user_data: RegisterRequest):
    """
    Register a new user
    
    - **email**: User's email address
    - **username**: Unique username
    - **password**: User's password (at least 6 characters)
    - **Additional fields**: Any other user data you want to store
    
    Returns:
        - 201: User registered successfully
        - 400: Invalid input data or email already in use
        - 500: Internal server error
    """
    try:
        logger.info(f"Registration attempt for email: {user_data.email}")
        
        # Validate input
        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
            
        if not user_data.email or "@" not in user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid email address is required"
            )
            
        if not user_data.username or len(user_data.username) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username must be at least 3 characters long"
            )
        
        # Convert Pydantic model to dict and remove None values
        user_dict = user_data.dict(exclude_none=True)
        
        # Extract password (required for auth)
        password = user_dict.pop('password')
        
        try:
            # Remove username and email from user_dict to avoid duplicate parameters
            if 'username' in user_dict:
                user_dict.pop('username')
            if 'email' in user_dict:
                user_dict.pop('email')
                
            # Register the user
            result = await auth_service.register_user(
                email=user_data.email,
                password=password,
                username=user_data.username,
                **user_dict
            )
            
            logger.info(f"Successfully registered user: {user_data.email}")
            
            return {
                "message": "User registered successfully",
                "user": {
                    "uid": result["uid"],
                    "email": result["email"],
                    "username": result["username"]
                },
                "token": result["token"]
            }
            
        except ValueError as e:
            logger.warning(f"Registration validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
            
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during registration: {str(e)}"
        )

@router.post("/auth/login", response_model=Dict[str, Any])
async def login_user(login_data: schemas.UserLogin):
    """
    Login user with email and password
    
    - **email**: User's email address
    - **password**: User's password
    """
    try:
        # Authenticate user
        result = await auth_service.login_user(
            email=login_data.email,
            password=login_data.password
        )
        
        return {
            "message": "Login successful",
            "user": {
                "uid": result["uid"],
                "email": result["email"],
                "username": result["username"]
            },
            "token": result["token"]
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login"
        )

@router.get("/auth/me", response_model=Dict[str, Any])
async def get_current_user_profile(current_user: dict = Depends(auth_scheme)):
    """
    Get the current user's profile
    """
    try:
        user_id = current_user.get('uid')
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
            
        profile = await auth_service.get_user_profile(user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
            
        return {
            "message": "User profile retrieved successfully",
            "user": profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the user profile"
        )
