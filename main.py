import os
import datetime
import traceback
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import motor.motor_asyncio
import uvicorn

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")

# Global client cache for connection reuse across function invocations
_cached_client = None

def get_mongo_client():
    """
    Get or create a cached MongoDB client for connection reuse.
    This prevents connection storming in serverless environments.
    """
    global _cached_client
    
    if _cached_client is None:
        print("🔄 Creating new MongoDB client...")
        _cached_client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=10000,         # 10 second connection timeout
            socketTimeoutMS=10000,          # 10 second socket timeout
            maxPoolSize=1,                  # Limit connections in serverless
            minPoolSize=0,                  # Don't maintain idle connections
            maxIdleTimeMS=10000,            # Close idle connections after 10s
            retryWrites=True,               # Retry failed writes
            w="majority"                    # Write concern
        )
        print("✅ MongoDB client created")
    
    return _cached_client

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]

# --- FastAPI App (NO LIFESPAN) ---
app = FastAPI(
    title="RAG Medical Query API",
    description="An API to get answers and contexts for medical questions.",
    version="0.1.0"
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
async def health_check():
    """
    Comprehensive health check endpoint with detailed diagnostics.
    """
    diagnostics = {
        "status": "unknown",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mongo_uri_set": bool(MONGO_URI),
        "mongo_uri_preview": MONGO_URI[:30] + "..." if MONGO_URI else None,
        "client_cached": _cached_client is not None
    }
    
    if not MONGO_URI:
        diagnostics["status"] = "error"
        diagnostics["error"] = "MONGO_URI not set"
        print(f"❌ Health check failed: MONGO_URI not set")
        return JSONResponse(status_code=503, content=diagnostics)
    
    try:
        print("🏥 Running health check...")
        client = get_mongo_client()
        
        # Try to ping MongoDB with timeout
        await client.admin.command('ping')
        
        diagnostics["status"] = "ok"
        diagnostics["message"] = "MongoDB connection successful"
        print(f"✅ Health check passed")
        return JSONResponse(status_code=200, content=diagnostics)
        
    except Exception as e:
        diagnostics["status"] = "error"
        diagnostics["error"] = str(e)
        diagnostics["error_type"] = type(e).__name__
        diagnostics["traceback"] = traceback.format_exc()
        print(f"❌ Health check failed with exception: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=503, content=diagnostics)

@app.post("/query", response_model=QueryResponse)
async def get_rag_response(query_request: QueryRequest):
    """
    Accepts a medical query and returns a generated answer with logging.
    """
    print(f"📝 Received query: '{query_request.query}' with top_k={query_request.top_k}")

    if not MONGO_URI:
        error_detail = {
            "error": "Service Unavailable",
            "message": "MONGO_URI not configured"
        }
        print(f"❌ MONGO_URI not set")
        return JSONResponse(status_code=503, content=error_detail)

    try:
        # Get MongoDB client
        client = get_mongo_client()
        db = client.event
        log_collection = db.rag_logs
        
        print("🔍 Processing query...")
        
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

        # --- Log Successful Response ---
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "request_query": query_request.query,
            "request_top_k": query_request.top_k,
            "response_answer": response.answer,
            "response_contexts": response.contexts,
            "status": "success"
        }
        print("💾 Storing log to MongoDB...")
        
        await log_collection.insert_one(log_entry)
        print("✅ Log stored successfully")

        return response

    except Exception as e:
        error_msg = f"CRITICAL ERROR processing request: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"❌ {error_msg}")
        print(f"   Traceback:\n{error_trace}")
        
        # --- Try to Log Error ---
        try:
            client = get_mongo_client()
            db = client.event
            log_collection = db.rag_logs
            
            error_log_data = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "request_query": query_request.query,
                "request_top_k": query_request.top_k,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "error_traceback": error_trace,
                "status": "error"
            }
            await log_collection.insert_one(error_log_data)
            print("💾 Error logged to database")
        except Exception as log_e:
            print(f"❌ Failed to log error: {str(log_e)}")

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

# --- Debug endpoint ---
@app.get("/debug/env")
async def debug_environment():
    """
    Debug endpoint to check environment variables
    """
    return {
        "mongo_uri_set": bool(MONGO_URI),
        "mongo_uri_length": len(MONGO_URI) if MONGO_URI else 0,
        "mongo_uri_preview": MONGO_URI[:50] + "..." if MONGO_URI and len(MONGO_URI) > 50 else MONGO_URI,
        "client_cached": _cached_client is not None,
        "python_version": os.sys.version,
    }

# --- For Local Development Only ---
if __name__ == "__main__":
    print("--- Starting local server ---")
    if not MONGO_URI:
        print("⚠️ WARNING: MONGO_URI is not set")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
