import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import motor.motor_asyncio  # Import the async MongoDB driver
import datetime              # To timestamp our logs
import os                    # To handle environment variables

# --- Configuration ---
# Load the MONGO_URI from Vercel's Environment Variables
MONGO_URI = os.getenv("MONGO_URI")

# --- Database Connection ---
client = None
db = None
log_collection = None

if not MONGO_URI:
    print("FATAL ERROR: MONGO_URI environment variable is not set.")
else:
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        db = client.event  # Your database is named 'event'
        log_collection = db.rag_logs  # Collection named 'rag_logs'
        print("Attempting to connect to MongoDB...")
        # We will verify the connection in the health check
    except Exception as e:
        print(f"ERROR: Could not initialize MongoDB client: {e}")
        client = None
        log_collection = None

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

# --- FastAPI App ---
app = FastAPI(
    title="RAG Medical Query API",
    description="An API to get answers and contexts for medical questions.",
    version="0.1.0"
)

# --- Endpoints ---

@app.get("/", include_in_schema=False)
async def health_check():
    """
    Simple health check to verify database connection.
    Visit this endpoint (/) in your browser.
    """
    if client is None or log_collection is None:
        raise HTTPException(
            status_code=503,
            detail="Mongo client not initialized. Check MONGO_URI env var."
        )
    
    try:
        # 'ping' is a lightweight command to check auth and connection
        await client.admin.command('ping')
        return {"status": "ok", "message": "Successfully connected to MongoDB."}
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to MongoDB: {e}"
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
        # Raise an error immediately so the user knows the service is down
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Cannot connect to log database."
        )

    try:
        # --- RAG Model Integration ---
        # TODO: Replace this placeholder logic with your actual RAG model call

        placeholder_contexts = [
            f"Placeholder context 1 for query: '{request.query}'",
            f"Placeholder context 2 (top_k was {request.top_k})",
            "Placeholder context 3: Always consult a medical professional."
        ]
        placeholder_answer = f"This is a placeholder answer for '{request.query}'. The RAG model is not yet integrated."

        response = QueryResponse(
            answer=placeholder_answer,
            contexts=placeholder_contexts
        )

        # --- Log Successful Response ---
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
        # --- FIXED ERROR HANDLING ---
        # We DO NOT try to log to MongoDB here, as that might be the 
        # very thing that failed. We print to Vercel logs instead.
        
        print(f"CRITICAL ERROR processing request: {e}")

        # Log the error details to the Vercel console
        error_log_data = {
            "request_query": request.query,
            "request_top_k": request.top_k,
            "error_message": str(e),
            "status": "error"
        }
        print(f"Failed to process request. Error data: {error_log_data}")

        # Raise the HTTP Exception
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred. Check logs. Error: {str(e)}"
        )

# --- To Run This Server Locally ---
if __name__ == "__main__":
    # This block allows you to run the app directly with "python main.py"
    # Uvicorn is the server that runs the app
    
    # For local running, you MUST set the env var in your terminal first:
    # export MONGO_URI="your_full_mongo_uri_string_here"
    # Or on Windows:
    # set MONGO_URI="your_full_mongo_uri_string_here"
    
    print("--- Starting local server ---")
    if not MONGO_URI:
        print("WARNING: MONGO_URI is not set. Database will not connect.")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
