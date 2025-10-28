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

def get_mongo_client():
    """
    Create a NEW MongoDB client for each request.
    DO NOT cache Motor clients in serverless - they get attached to closed event loops.
    """
    print("🔄 Creating fresh MongoDB client for this request...")
    client = motor.motor_asyncio.AsyncIOMotorClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000,
        maxPoolSize=1,
        minPoolSize=0
    )
    print("✅ MongoDB client created")
    return client

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
    }
    
    if not MONGO_URI:
        diagnostics["status"] = "error"
        diagnostics["error"] = "MONGO_URI not set"
        print(f"❌ Health check failed: MONGO_URI not set")
        return JSONResponse(status_code=503, content=diagnostics)
    
    client = None
    try:
        print("🏥 Running health check...")
        client = get_mongo_client()
        
        # Try to ping MongoDB
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
        print(f"❌ Health check failed: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=503, content=diagnostics)
    
    finally:
        # CRITICAL: Close the client after each request in serverless
        if client:
            client.close()
            print("🔒 MongoDB client closed")

@app.post("/query")
async def get_rag_response(query_request: QueryRequest):
    """
    Accepts a medical query and returns a generated answer with logging.
    """
    print(f"📝 Received query: '{query_request.query}' with top_k={query_request.top_k}")

    if not MONGO_URI:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service Unavailable",
                "message": "MONGO_URI not configured"
            }
        )

    client = None
    try:
        # Create fresh client for this request
        client = get_mongo_client()
        db = client.event
        log_collection = db.rag_logs
        
        print("🔍 Processing query...")
        
        # --- RAG Model Integration ---
        placeholder_contexts = [
            f"Placeholder context 1 for query: '{query_request.query}'",
            f"Placeholder context 2 (top_k was {query_request.top_k})",
        ]
        placeholder_answer = f"This is a placeholder answer for '{query_request.query}'."

        response = {
            "answer": placeholder_answer,
            "contexts": placeholder_contexts
        }

        # --- Log Successful Response ---
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "request_query": query_request.query,
            "request_top_k": query_request.top_k,
            "response_answer": response["answer"],
            "response_contexts": response["contexts"],
            "status": "success"
        }
        print("💾 Storing log to MongoDB...")
        
        await log_collection.insert_one(log_entry)
        print("✅ Log stored successfully")

        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        error_msg = f"ERROR processing request: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"❌ {error_msg}")
        print(f"   Traceback:\n{error_trace}")
        
        # Try to log error
        try:
            if client:
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

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "type": type(e).__name__,
                "message": str(e),
                "traceback": error_trace
            }
        )
    
    finally:
        # CRITICAL: Always close client in serverless
        if client:
            client.close()
            print("🔒 MongoDB client closed")

# --- Debug endpoint ---
@app.get("/debug/env")
async def debug_environment():
    """
    Debug endpoint
    """
    return {
        "mongo_uri_set": bool(MONGO_URI),
        "mongo_uri_length": len(MONGO_URI) if MONGO_URI else 0,
        "mongo_uri_preview": MONGO_URI[:50] + "..." if MONGO_URI and len(MONGO_URI) > 50 else MONGO_URI,
        "python_version": os.sys.version,
    }

# --- Local Development ---
if __name__ == "__main__":
    print("--- Starting local server ---")
    if not MONGO_URI:
        print("⚠️ WARNING: MONGO_URI is not set")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
