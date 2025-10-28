import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import motor.motor_asyncio  # Import the async MongoDB driver
import datetime             # To timestamp our logs
import os                   # To handle environment variables

# --- Configuration ---
# IMPORTANT: You should store your MONGO_URI in Vercel's Environment Variables
# I am using os.getenv as a best practice.
# For local testing, you can set it, or it will use the hardcoded fallback.
MONGO_URI = os.getenv(
    "MONGO_URI", "mongodb+srv://Ahsan:Ahsan2006@event.dpqpsgj.mongodb.net/")

# --- Database Connection ---
try:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db = client.event  # Your database is named 'event'
    log_collection = db.rag_logs  # We'll create/use a collection named 'rag_logs'
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"ERROR: Could not connect to MongoDB: {e}")
    client = None
    log_collection = None

# --- Pydantic Models ---
# This model defines the structure of the incoming request body


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# This model defines the structure of the successful response body


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]


# --- FastAPI App ---
app = FastAPI(
    title="RAG Medical Query API",
    description="An API to get answers and contexts for medical questions.",
    version="0.1.0"
)


@app.post("/query", response_model=QueryResponse)
async def get_rag_response(request: QueryRequest):
    """
    Accepts a medical query and returns a generated answer along with
    the context snippets used to generate that answer.

    All requests, responses, and errors are logged to MongoDB.
    """

    print(f"Received query: '{request.query}' with top_k={request.top_k}")

    if log_collection is None:
        print("Warning: MongoDB not connected. Skipping logging.")

    try:
        # --- RAG Model Integration ---
        # TODO: Replace this placeholder logic with your actual RAG model call

        # Simulating context retrieval
        placeholder_contexts = [
            f"Placeholder context 1 for query: '{request.query}'",
            f"Placeholder context 2 (top_k was {request.top_k})",
            "Placeholder context 3: Always consult a medical professional."
        ]

        # Simulating answer generation
        placeholder_answer = f"This is a placeholder answer for '{request.query}'. The RAG model is not yet integrated."

        # Create the response object
        response = QueryResponse(
            answer=placeholder_answer,
            contexts=placeholder_contexts
        )

        # --- Log Successful Response ---
        if log_collection is not None:
            log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "request_query": request.query,
                "request_top_k": request.top_k,
                "response_answer": response.answer,
                "response_contexts": response.contexts,
                "status": "success"
            }
            print("Storing successful request log to MongoDB...")
            await log_collection.insert_one(log_entry)
            print("...Log stored successfully.")

        return response

    except Exception as e:
        # In a real app, you'd log the error e
        print(f"Error processing request: {e}")

        # --- Log Error ---
        if log_collection is not None:
            error_log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "request_query": request.query,
                "request_top_k": request.top_k,
                "error_message": str(e),
                "status": "error"
            }
            print("Storing error log to MongoDB...")
            await log_collection.insert_one(error_log_entry)
            print("...Error log stored.")

        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred."
        )

# --- To Run This Server Locally ---
# 1. Make sure you have fastapi and uvicorn installed:
#    pip install fastapi uvicorn motor
# 2. Run the server from your terminal:
#    uvicorn main:app --reload
#
# 3. You can then send a POST request.
#
#    --- Option 1: Using curl ---
#    curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"What are the symptoms of diabetes?\", \"top_k\": 3}"
#
#    --- Option 2: Using Postman ---
#    - Method: POST
#    - URL: http://127.0.0.1:8000/query
#    - Body tab -> select 'raw' -> select 'JSON'
#    - Paste: {"query": "What are...", "top_k": 3}
#

if __name__ == "__main__":
    # This block allows you to run the app directly with "python main.py"
    # Uvicorn is the server that runs the app
    uvicorn.run(app, host="127.0.0.1", port=8000)
