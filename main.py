import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import motor.motor_asyncio  # Import the async MongoDB driver
import datetime              # To timestamp our logs
import os                    # To handle environment variables
from contextlib import asynccontextmanager

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")

# --- Database Connection Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the MongoDB connection lifespan.
    It NOW ONLY CREATES THE CLIENT, IT DOES NOT PING.
    """
    if not MONGO_URI:
        print("FATAL ERROR: MONGO_URI environment variable is not set.")
        app.state.client = None
        app.state.db = None
        app.state.log_collection = None
        yield
        return
        
    print("Initializing MongoDB client...")
    try:
        # We ONLY create the client instance. We do NOT ping.
        # This will allow the app to start even if the DB is unreachable.
        app.state.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        app.state.db = app.state.client.event
        app.state.log_collection = app.state.db.rag_logs
        print("MongoDB client initialized.")
        
    except Exception as e:
        print(f"FATAL: Could not initialize MongoDB client: {e}")
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
    This will now be the FIRST place that actually tries to connect.
    """
    if request.app.state.client is None:
        raise HTTPException(
            status_code=503,
            detail="Mongo client not initialized. Check MONGO_URI env var or startup logs."
        )
    
    try:
        # The ping is now HERE.
        print("Health check: Pinging MongoDB...")
        await request.app.state.client.admin.command('ping')
        print("Health check: Ping successful.")
        return {"status": "ok", "message": "Successfully connected to MongoDB."}
    except Exception as e:
        # If the ping fails, we will now SEE THE ERROR.
        print(f"Health check FAILED: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to MongoDB. Error: {e}"
        )

@app.post("/query", response_model=QueryResponse)
async def get_rag_response(query_request: QueryRequest, request: Request):
    """
    Accepts a medical query and returns a generated answer.
    """
    log_collection = request.app.state.log_collection
    print(f"Received query: '{query_request.query}' with top_k={query_request.top_k}")

    if log_collection is None:
        print("Warning: MongoDB not connected. Skipping logging.")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Cannot connect to log database."
        )

    try:
        # ... (Placeholder logic remains the same) ...
        placeholder_contexts = [f"Context for '{query_request.query}'"]
        placeholder_answer = f"Answer for '{query_request.query}'"
        response = QueryResponse(answer=placeholder_answer, contexts=placeholder_contexts)

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
