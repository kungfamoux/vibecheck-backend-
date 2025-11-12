from firebase_admin import auth
from firebase_admin import db as firebase_db
from datetime import datetime
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class AuthService:
    @staticmethod
    async def register_user(email: str, password: str, username: str = None, **user_data) -> Dict:
        """
        Register a new user with Firebase Authentication and store additional data in Realtime Database
        
        Args:
            email: User's email address (required)
            password: User's password (required, will be hashed by Firebase)
            username: User's display name (optional)
            **user_data: Additional user data to store
            
        Returns:
            Dictionary containing user data and auth token
            
        Raises:
            ValueError: If user registration fails
        """
        # Clean up parameters to avoid duplicates
        if 'email' in user_data:
            user_data.pop('email')
            
        # If username is in user_data but not passed as parameter, use it from user_data
        if username is None and 'username' in user_data:
            username = user_data.pop('username')
            
        # If still no username, use the part before @ in email
        if username is None:
            username = email.split('@')[0]
            
        try:
            logger.info(f"Attempting to register user: {email}")
            
            # Validate input
            if not email or not password or not username:
                raise ValueError("Email, password, and username are required")
                
            # Check if email is already in use
            try:
                existing_user = auth.get_user_by_email(email)
                if existing_user:
                    raise ValueError("Email already in use")
            except auth.UserNotFoundError:
                pass  # This is expected - email is available
                
            # Create user in Firebase Auth
            try:
                user = auth.create_user(
                    email=email,
                    password=password,
                    display_name=username
                )
                logger.info(f"Successfully created auth user: {user.uid}")
            except Exception as auth_error:
                logger.error(f"Auth user creation failed: {str(auth_error)}")
                raise ValueError("Failed to create user account. Please try again.")
            
            # Prepare user data for Realtime Database
            user_profile = {
                'uid': user.uid,
                'email': email,
                'username': username,
                'created_at': datetime.utcnow().isoformat(),
                'is_active': True,
                **user_data
            }
            
            try:
                # Save user data to Realtime Database
                user_ref = firebase_db.reference(f'users/{user.uid}')
                user_ref.set(user_profile)
                logger.info(f"Successfully saved user data to Realtime Database: {user.uid}")
            except Exception as db_error:
                # If database save fails, clean up the auth user
                try:
                    auth.delete_user(user.uid)
                    logger.warning(f"Cleaned up auth user after database error: {user.uid}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up auth user: {str(cleanup_error)}")
                
                logger.error(f"Database save failed: {str(db_error)}")
                raise ValueError("Failed to save user data. Please try again.")
            
            # Generate custom token for the user
            custom_token = auth.create_custom_token(user.uid)
            
            return {
                'uid': user.uid,
                'email': user.email,
                'username': username,
                'token': custom_token,
                'profile': user_profile
            }
            
        except auth.EmailAlreadyExistsError:
            raise ValueError("Email already exists")
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            raise
    
    @staticmethod
    async def login_user(email: str, password: str) -> Dict:
        """
        Login user with email and password
        
        Args:
            email: User's email address
            password: User's password
            
        Returns:
            Dictionary containing user data and auth token
        """
        try:
            # In a real implementation, you would verify the email/password with Firebase Auth
            # and then generate a custom token or use Firebase client SDK on the frontend
            
            # For now, we'll just verify the user exists and return a mock response
            user = auth.get_user_by_email(email)
            
            # Get user data from Realtime Database
            user_ref = firebase_db.reference(f'users/{user.uid}')
            user_data = user_ref.get()
            
            if not user_data:
                raise ValueError("User data not found")
                
            # Generate custom token for the user
            custom_token = auth.create_custom_token(user.uid)
            
            return {
                'uid': user.uid,
                'email': user.email,
                'username': user_data.get('username', ''),
                'token': custom_token,
                'profile': user_data
            }
            
        except auth.UserNotFoundError:
            raise ValueError("Invalid email or password")
        except Exception as e:
            logger.error(f"Error logging in user: {e}")
            raise
    
    @staticmethod
    async def get_user_profile(uid: str) -> Optional[Dict]:
        """
        Get user profile from Realtime Database
        
        Args:
            uid: User ID
            
        Returns:
            User profile data or None if not found
        """
        try:
            user_ref = firebase_db.reference(f'users/{uid}')
            return user_ref.get()
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None

# Create a global instance of the auth service
auth_service = AuthService()
