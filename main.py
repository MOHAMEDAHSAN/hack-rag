import os
import datetime
import traceback
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import motor.motor_asyncio
import uvicorn

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")

# Global variable to store startup errors
startup_error = None

# --- MongoDB Connection Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the MongoDB connection lifespan for async operations.
    """
    global startup_error
    
    if not MONGO_URI:
        error_msg = "FATAL ERROR: MONGO_URI environment variable is not set."
        print(error_msg)
        startup_error = error_msg
        app.state.client = None
        app.state.db = None
        app.state.log_collection = None
        yield
        return
        
    print("Initializing MongoDB ASYNC client with Motor...")
    try:
        # Use Motor's AsyncIOMotorClient for async operations
        app.state.client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000
        )
        
        # Ping on startup to confirm connection
        await app.state.client.admin.command('ping')
        
        app.state.db = app.state.client.event
        app.state.log_collection = app.state.db.rag_logs
        print("✅ MongoDB ASYNC client initialized and ping successful.")
        startup_error = None
        
    except Exception as e:
        error_msg = f"FATAL: Could not connect to MongoDB on startup: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        startup_error = error_msg
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
    lifespan=lifespan
)

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch ALL unhandled exceptions and return detailed error info.
    """
    error_detail = {
        "error": "Internal Server Error",
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "path": str(request.url),
        "method": request.method
    }
    
    print(f"❌ UNHANDLED EXCEPTION:")
    print(f"   Type: {type(exc).__name__}")
    print(f"   Message: {str(exc)}")
    print(f"   Traceback:\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=error_detail
    )

# --- Validation Error Handler ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors with detailed info.
    """
    error_detail = {
        "error": "Validation Error",
        "message": "Request validation failed",
        "details": exc.errors(),
        "body": exc.body if hasattr(exc, 'body') else None
    }
    
    print(f"❌ VALIDATION ERROR: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content=error_detail
    )

# --- Endpoints ---

@app.get("/")
async def health_check(request: Request):
    """
    Comprehensive health check endpoint with detailed diagnostics.
    """
    global startup_error
    
    diagnostics = {
        "status": "unknown",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mongo_uri_set": bool(MONGO_URI),
        "mongo_uri_preview": MONGO_URI[:30] + "..." if MONGO_URI else None,
        "client_initialized": request.app.state.client is not None,
        "db_initialized": request.app.state.db is not None,
        "collection_initialized": request.app.state.log_collection is not None,
        "startup_error": startup_error
    }
    
    # Test MongoDB connection
    if request.app.state.client is None:
        diagnostics["status"] = "error"
        diagnostics["error"] = "MongoDB client not initialized"
        print(f"❌ Health check failed: {diagnostics}")
        return JSONResponse(status_code=503, content=diagnostics)
    
    try:
        # Try to ping MongoDB
        await request.app.state.client.admin.command('ping')
        diagnostics["status"] = "ok"
        diagnostics["message"] = "All systems operational"
        print(f"✅ Health check passed")
        return JSONResponse(status_code=200, content=diagnostics)
        
    except Exception as e:
        diagnostics["status"] = "error"
        diagnostics["error"] = str(e)
        diagnostics["traceback"] = traceback.format_exc()
        print(f"❌ Health check failed with exception: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=503, content=diagnostics)

@app.post("/query", response_model=QueryResponse)
async def get_rag_response(query_request: QueryRequest, request: Request):
    """
    Accepts a medical query and returns a generated answer with logging.
    """
    log_collection = request.app.state.log_collection
    print(f"📝 Received query: '{query_request.query}' with top_k={query_request.top_k}")

    if log_collection is None:
        error_detail = {
            "error": "Service Unavailable",
            "message": "MongoDB not connected. Cannot log requests.",
            "mongo_uri_set": bool(MONGO_URI),
            "startup_error": startup_error
        }
        print(f"❌ MongoDB not available: {error_detail}")
        return JSONResponse(status_code=503, content=error_detail)

    try:
        # --- RAG Model Integration (Replace with your actual model) ---
        placeholder_contexts = [
            f"Placeholder context 1 for query: '{query_request.query}'",
            f"Placeholder context 2 (top_k was {query_request.top_k})",
        ]
        placeholder_answer = f"This is a placeholder answer for '{query_request.query}'."

        response = QueryResponse(
            answer=placeholder_answer,
            contexts=placeholder_contexts
        )

        # --- Log Successful Response (ASYNC with await) ---
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "request_query": query_request.query,
            "request_top_k": query_request.top_k,
            "response_answer": response.answer,
            "response_contexts": response.contexts,
            "status": "success"
        }
        print("💾 Storing successful request log to MongoDB (async)...")
        
        # CRITICAL: Use await for async Motor operations
        await log_collection.insert_one(log_entry)
        print("✅ Log stored successfully.")

        return response

    except Exception as e:
        error_msg = f"CRITICAL ERROR processing request: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"❌ {error_msg}")
        print(f"   Traceback:\n{error_trace}")
        
        # --- Log Error (ASYNC with await) ---
        try:
            error_log_data = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "request_query": query_request.query,
                "request_top_k": query_request.top_k,
                "error_message": str(e),
                "error_traceback": error_trace,
                "status": "error"
            }
            await log_collection.insert_one(error_log_data)
            print("💾 Error logged to database")
        except Exception as log_e:
            print(f"❌ Failed to even log the error: {str(log_e)}")

        # Return detailed error
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "type": type(e).__name__,
                "message": str(e),
                "traceback": error_trace
            }
        )

# --- Debug endpoint to check environment ---
@app.get("/debug/env")
async def debug_environment():
    """
    Debug endpoint to check environment variables (remove in production!)
    """
    return {
        "mongo_uri_set": bool(MONGO_URI),
        "mongo_uri_length": len(MONGO_URI) if MONGO_URI else 0,
        "mongo_uri_preview": MONGO_URI[:50] + "..." if MONGO_URI and len(MONGO_URI) > 50 else MONGO_URI,
        "python_version": os.sys.version,
        "environment_vars": list(os.environ.keys())
    }

# --- For Local Development Only ---
if __name__ == "__main__":
    print("--- Starting local server ---")
    if not MONGO_URI:
        print("⚠️ WARNING: MONGO_URI is not set. Database will not connect.")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
