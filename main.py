from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import firebase_admin
from firebase_admin import credentials, auth
import logging
from typing import List
import os
from dotenv import load_dotenv

# Import API routers
from api.endpoints import router as api_router
from api.auth_endpoints import router as auth_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Firebase Admin
SERVICE_ACCOUNT_FILE = "recommend-16f0e-firebase-adminsdk-fbsvc-235e14bc49.json"

# Initialize Firebase Admin if not already initialized
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Firebase Admin: {e}")
        raise

# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis Recommendation API",
    description="API for content recommendations based on sentiment analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",  # React default port
    "http://localhost:8000",
    "http://127.0.0.1:3000",  # Alternative localhost
    "https://your-production-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router)
app.include_router(auth_router)

# Security scheme for API documentation
security = HTTPBearer()

# Health check endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Sentiment Analysis Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle global exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "detail": "An unexpected error occurred. Please try again later.",
        "error": str(exc)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize ML service
        from services import ml_service
        logger.info("ML service initialized")
        
        # Initialize recommendation engine
        from services import recommendation_engine
        logger.info("Recommendation engine initialized")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")

# For debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
