# Sentiment Analysis-Driven Recommendation System

A backend service that provides content recommendations based on sentiment analysis of user input and interactions.

## Features

- **User Authentication**: Secure authentication using Firebase Authentication
- **Sentiment Analysis**: Analyzes user comments and feedback to determine sentiment
- **Content Recommendations**: Provides personalized content recommendations based on user mood and preferences

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Firebase project and get the service account credentials
4. Update the `.env` file with your Firebase credentials

## Running the Application

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

- `POST /analyze`: Analyze sentiment of user input
- `POST /recommend`: Get content recommendations based on user input
- `GET /health`: Health check endpoint

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY=your-private-key
FIREBASE_CLIENT_EMAIL=your-client-email
FIREBASE_CLIENT_ID=your-client-id
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```
