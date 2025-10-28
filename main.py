import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional
import pymongo  # <-- IMPORT PYMONGO, NOT MOTOR
import datetime
import os
from contextlib import asynccontextmanager

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")

# --- Database Connection Lifespan ---
# We still use lifespan to create the client, but it's now a SYNC client.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the MongoDB connection lifespan.
    """
    if not MONGO_URI:
        print("FATAL ERROR: MONGO_URI environment variable is not set.")
        app.state.client = None
        app.state.db = None
        app.state.log_collection = None
        yield
        return
        
    print("Initializing MongoDB SYNC client...")
    try:
        # --- THIS IS THE BIG CHANGE ---
        # We are using the standard, synchronous Pymongo client
        app.state.client = pymongo.MongoClient(MONGO_URI)
        
        # Ping on startup to confirm connection.
        # This will raise an error if it fails, which is what we want.
        app.state.client.admin.command('ping')
        
        app.state.db = app.state.client.event
        app.state.log_collection = app.state.db.rag_logs
        print("MongoDB SYNC client initialized and ping successful.")
        
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
def health_check(request: Request):  # <-- Note: no async
    """
    Simple health check.
    """
    if request.app.state.client is None:
        raise HTTPException(
            status_code=503,
            detail="Mongo client not initialized. Check startup logs."
        )
    
    # We don't need to ping again, startup already did.
    return {"status": "ok", "message": "Successfully connected to MongoDB (sync client)."}

@app.post("/query", response_model=QueryResponse)
def get_rag_response(query_request: QueryRequest, request: Request): # <-- Note: no async
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
        # --- RAG Model Integration ---
        placeholder_contexts = [
            f"Placeholder context 1 for query: '{query_request.query}'",
            f"Placeholder context 2 (top_k was {query_request.top_k})",
        ]
        placeholder_answer = f"This is a placeholder answer for '{query_request.query}'."

        response = QueryResponse(
            answer=placeholder_answer,
            contexts=placeholder_contexts
        )

        # --- Log Successful Response (SYNC) ---
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "request_query": query_request.query,
            "request_top_k": query_request.top_k,
            "response_answer": response.answer,
            "response_contexts": response.contexts,
            "status": "success"
        }
        print("Storing successful request log to MongoDB (sync)...")
        # --- NO AWAIT ---
        log_collection.insert_one(log_entry)
        print("...Log stored successfully.")

        return response

    except Exception as e:
        print(f"CRITICAL ERROR processing request: {e}")
        # --- Log Error (SYNC) ---
        try:
            error_log_data = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "request_query": query_request.query,
                "request_top_k": query_request.top_k,
                "error_message": str(e),
                "status": "error"
            }
            log_collection.insert_one(error_log_data)
        except Exception as log_e:
            print(f"Failed to even log the error: {log_e}")

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
