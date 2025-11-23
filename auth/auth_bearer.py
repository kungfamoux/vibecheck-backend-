from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization code.",
            )
            
        if not credentials.scheme == "Bearer":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authentication scheme.",
            )
            
        token = self.verify_jwt(credentials.credentials)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid token or expired token.",
            )
            
        return token

    def verify_jwt(self, jwt_token: str) -> dict:
        """Verify the JWT token with Firebase Admin SDK"""
        try:
            # First try to verify as ID token
            try:
                decoded_token = auth.verify_id_token(jwt_token)
                return decoded_token
            except ValueError as e:
                # If it's not an ID token, try to verify as a custom token
                if 'verify_id_token() expects an ID token' in str(e):
                    try:
                        # Get the user by the custom token
                        user = auth.get_user_by_phone_number(jwt_token)  # or any other identifier
                        if user:
                            return {'uid': user.uid, 'email': user.email}
                    except Exception:
                        pass
                logger.error(f"Token verification error: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error during token verification: {e}")
            return None

# Create an instance of JWTBearer to use as a dependency
auth_scheme = JWTBearer()
