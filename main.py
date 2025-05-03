import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from chat_router import router as chat_router

# Try simple_rag first, then fall back to rag
try:
    from simple_rag import load_data, get_data_summary
    logger = logging.getLogger(__name__)
    logger.info("Using simple_rag module in main (no sentence-transformers)")
except ImportError:
    from rag import load_data, get_data_summary
    logger = logging.getLogger(__name__)
    logger.info("Using full rag module in main (with sentence-transformers)")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Data-Enhanced Chatbot",
    description="A chatbot that answers questions from specific data or general knowledge",
    version="1.0.0",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the chat router
app.include_router(chat_router, prefix="/chat", tags=["Chat"])

@app.get("/")
def read_root():
    """Root endpoint that shows the API is running."""
    return {
        "status": "online",
        "message": "Data-Enhanced Chatbot API is running",
        "usage": "Send your queries to /chat/message",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/data/summary")
def data_summary():
    """Get a summary of the loaded data."""
    return {"summary": get_data_summary()}

@app.post("/data/reload")
def reload_data():
    """Reload the data from the CSV file."""
    success = load_data()
    if success:
        return {"status": "success", "message": "Data reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload data")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    
    # Log startup information
    logger.info(f"Starting server on port {port}")
    logger.info("Loading data...")
    load_data()
    
    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)