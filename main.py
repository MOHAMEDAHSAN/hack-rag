import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import motor.motor_asyncio  # Import the async MongoDB driver
import datetime              # To timestamp our logs
import os                    # To handle environment variables
from contextlib import asynccontextmanager

# --- Configuration ---
# Load the MONGO_URI from Vercel's Environment Variables
MONGO_URI = os.getenv("MONGO_URI")

# --- Database Connection Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the MongoDB connection lifespan.
    The connection is established on startup and closed on shutdown.
    """
    if not MONGO_URI:
        print("FATAL ERROR: MONGO_URI environment variable is not set.")
        app.state.client = None
        app.state.db = None
        app.state.log_collection = None
        yield
        # No connection to close
        return
        
    print("Connecting to MongoDB...")
    try:
        # --- THIS IS THE CORRECTED LINE ---
        app.state.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        
        app.state.db = app.state.client.event
        app.state.log_collection = app.state.db.rag_logs
        
        # Test the connection with a ping
        await app.state.client.admin.command('ping')
        print("Successfully connected to MongoDB.")
        
    except Exception as e:
        print(f"FATAL: Could not connect to MongoDB on startup: {e}")
        app.state.client = None
        app.state.db = None
        app.state.log_collection = None

    # --- Application is now running ---
    yield
    # --- Application is shutting down ---
    
    if app.state.client:
        print("Closing MongoDB connection...")
        app.state.client.close()


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
    version="0.1.0",
    lifespan=lifespan  # Manages startup/shutdown
)

# --- Endpoints ---

@app.get("/", include_in_schema=False)
async def health_check(request: Request):
    """
    Simple health check to verify database connection.
    It accesses the client from the app's state.
    """
    if request.app.state.client is None:
        raise HTTPException(
            status_code=503,
            detail="Mongo client not initialized. Check MONGO_URI env var or startup logs."
        )
    
    try:
        # Ping again to be sure
        await request.app.state.client.admin.command('ping')
        return {"status": "ok", "message": "Successfully connected to MongoDB."}
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to MongoDB: {e}"
        )

@app.post("/query", response_model=QueryResponse)
async def get_rag_response(query_request: QueryRequest, request: Request):
    """
    Accepts a medical query and returns a generated answer along with
    the context snippets used to generate that answer.
    """
    # Get the log_collection from the application state
    log_collection = request.app.state.log_collection

    print(f"Received query: '{query_request.query}' with top_k={query_request.top_k}")

    if log_collection is None:
        print("Warning: MongoDB not connected. Skipping logging.")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Cannot connect to log database."
        )

    try:
        # --- RAG Model Integration ---
        # TODO: Replace this placeholder logic with your actual RAG model call

        placeholder_contexts = [
            f"Placeholder context 1 for query: '{query_request.query}'",
            f"Placeholder context 2 (top_k was {query_request.top_k})",
            "Placeholder context 3: Always consult a medical professional."
        ]
        placeholder_answer = f"This is a placeholder answer for '{query_request.query}'. The RAG model is not yet integrated."

        response = QueryResponse(
            answer=placeholder_answer,
            contexts=placeholder_contexts
        )

        # --- Log Successful Response ---
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "request_query": query_request.query,
            "request_top_k": query_request.top_k,
            "response_answer": response.answer,
            "response_contexts": response.contexts,
            "status": "success"
        }
        print("Storing successful request log to MongoDB...")
        await log_collection.insert_one(log_entry)
        print("...Log stored successfully.")

        return response

    except Exception as e:
        print(f"CRITICAL ERROR processing request: {e}")
        error_log_data = {
            "request_query": query_request.query,
            "request_top_k": query_request.top_k,
            "error_message": str(e),
            "status": "error"
        }
        print(f"Failed to process request. Error data: {error_log_data}")

        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred. Check logs. Error: {str(e)}"
        )

# --- To Run This Server Locally ---
if __name__ == "__main__":
    print("--- Starting local server ---")
    if not MONGO_URI:
        print("WARNING: MONGO_URI is not set. Database will not connect.")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
