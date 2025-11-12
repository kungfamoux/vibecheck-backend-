from firebase_admin import firestore, db as firebase_db
import firebase_admin
from firebase_admin import credentials, auth
import os

# Initialize Firebase Admin with the service account
SERVICE_ACCOUNT_FILE = "recommend-16f0e-firebase-adminsdk-fbsvc-235e14bc49.json"

# Initialize Firebase Admin if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://recommend-16f0e-default-rtdb.firebaseio.com/'  # Replace with your Realtime Database URL
    })

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
