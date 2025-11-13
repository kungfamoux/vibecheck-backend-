import os
import firebase_admin
import json
from firebase_admin import credentials, firestore, db as firebase_db, auth
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Firebase Admin if not already initialized
if not firebase_admin._apps:
    try:
        # Get required environment variables
        firebase_config = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
            "universe_domain": "googleapis.com"
        }

        # Verify required fields
        required_fields = ["project_id", "private_key", "client_email"]
        missing_fields = [field for field in required_fields if not firebase_config.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required Firebase config fields: {', '.join(missing_fields)}")

        # Initialize Firebase
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://recommend-16f0e-default-rtdb.firebaseio.com/'
        })
        print("Firebase Admin SDK initialized successfully")
        
    except Exception as e:
        print(f"Error initializing Firebase Admin: {str(e)}")
        print("Firebase configuration:", {k: v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v 
                                       for k, v in firebase_config.items() if k != 'private_key'})
        raise

# Get Firestore client
db = firestore.client()

# Get Realtime Database reference
def get_realtime_db():
    return firebase_db.reference()

# Collection/Node names
USERS_COLLECTION = "users"
USER_PROFILES_COLLECTION = "user_profiles"
CONTENT_COLLECTION = "content"
INTERACTIONS_COLLECTION = "interactions"
SENTIMENT_COLLECTION = "sentiment"

# Initialize Firestore collections
users_ref = db.collection(USERS_COLLECTION)
user_profiles_ref = db.collection(USER_PROFILES_COLLECTION)
content_ref = db.collection(CONTENT_COLLECTION)
interactions_ref = db.collection(INTERACTIONS_COLLECTION)
sentiment_ref = db.collection(SENTIMENT_COLLECTION)

# Realtime Database references
def get_users_rtd():
    return get_realtime_db().child('users')

def get_content_rtd():
    return get_realtime_db().child('content')

def get_interactions_rtd():
    return get_realtime_db().child('interactions')

def get_sentiment_rtd():
    return get_realtime_db().child('sentiment')
