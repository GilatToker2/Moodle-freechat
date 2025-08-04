"""
🎓 Academic Content Processing API

📖 API Documentation: http://localhost:8080/docs

🔑 Required old_config.py settings:
- STORAGE_CONNECTION_STRING
- CONTAINER_NAME ("course")
- AZURE_OPENAI_API_KEY
- VIDEO_INDEXER_ACCOUNT_ID
- SEARCH_SERVICE_NAME
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
from Config.logging_config import setup_logging
from Source.Services.free_chat import RAGSystem


# Initialize logger
logger = setup_logging()

# Convenience function for backward compatibility
def debug_log(message):
    """Write debug message using proper logging"""
    logger.debug(message)


# Initialize FastAPI app
app = FastAPI(title="Chat Service API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_system = RAGSystem()

# ================================
# 📋 RESPONSE MODELS
# ================================

class ErrorResponse(BaseModel):
    detail: str

class FreeChatRequest(BaseModel):
    conversation_id: str
    conversation_history: List[Dict[str, Any]]
    course_id: str
    user_message: str
    stage: str
    source_id: Optional[str] = None

class FreeChatResponse(BaseModel):
    conversation_id: str
    course_id: str
    user_message: str
    stage: str
    final_answer: str
    sources: List[Dict[str, Any]]
    timestamp: str
    success: bool


# ================================
# 🏠 ROOT & HEALTH ENDPOINTS
# ================================

@app.get("/", tags=["System"])
async def root():
    """Home page - Search & Chat Service information"""
    return {
        "message": "🔍💬 Search & Chat Service",
        "version": "1.0.0",
        "status": "Active",
        "functions": [
            "💬 /free-chat - RAG-based conversational AI",
            "🔍 /search - Advanced content search",
            "📊 /index/status - Check index status"
        ],
        "docs_url": "/docs"
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "search-chat-service"}

# ================================
# 💬 FREE CHAT ENDPOINTS
# ================================

@app.post(
    "/free-chat",
    response_model=FreeChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Free chat failed"}
    },
    tags=["Free Chat"]
)
async def free_chat_endpoint(request: FreeChatRequest):
    """
    💬 Free Chat with RAG-based Responses

    **Function Description:**
    Provides conversational AI responses based on course content using RAG (Retrieval-Augmented Generation).

    **What to Expect:**
    • Searches relevant content from the knowledge base using semantic search
    • Considers conversation history for context
    • Generates responses based only on indexed course content
    • Filters by course_id and optionally by source_id
    • Returns both the answer and source information

    **Request Body Example:**
    ```json
    {
        "conversation_id": "demo-123",
        "conversation_history": [
            {"role": "user", "content": "Hello", "timestamp": "2025-01-01T10:00:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2025-01-01T10:00:01"}
        ],
        "course_id": "Discrete_mathematics",
        "user_message": "מה זה יחס שקילות?",
        "stage": "regular_chat",
        "source_id": "2"
    }
    ```

    **Parameters:**
    - **conversation_id**: Unique identifier for the conversation
    - **conversation_history**: List of previous messages for context
    - **course_id**: Course identifier to filter relevant content
    - **user_message**: Current user question/message
    - **stage**: User stage (regular_chat/quiz_mode/presentation_discussion)
    - **source_id**: Optional - filter to specific source (video/document)

    **Returns:**
    - All input fields preserved
    - **final_answer**: RAG-based response in Hebrew
    - **sources**: Detailed information about sources used
    - **timestamp**: Response generation time
    - **success**: Boolean indicating operation success
    """
    try:
        logger.info(f"💬 Free chat request: {request.user_message} (course: {request.course_id})")

        # Validate required fields
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id is required")
        if not request.course_id:
            raise HTTPException(status_code=400, detail="course_id is required")
        if not request.user_message:
            raise HTTPException(status_code=400, detail="user_message is required")
        if not request.stage:
            raise HTTPException(status_code=400, detail="stage is required")

        # Call RAG system
        result = rag_system.generate_answer(
            conversation_id=request.conversation_id,
            conversation_history=request.conversation_history,
            course_id=request.course_id,
            user_message=request.user_message,
            stage=request.stage,
            source_id=request.source_id
        )

        # Log result
        if result['success']:
            logger.info(f"✅ Generated answer for: {request.user_message}")
        else:
            logger.warning(f"❌ Failed to generate answer: {result.get('error', 'Unknown error')}")

        # Return cleaned response without conversation_history
        return FreeChatResponse(
            conversation_id=result['conversation_id'],
            course_id=result['course_id'],
            user_message=result['user_message'],
            stage=result['stage'],
            final_answer=result['final_answer'],
            sources=result['sources'],
            timestamp=result['timestamp'],
            success=result['success']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in free chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Free chat failed: {str(e)}")



# ================================
# 🚀 SERVER STARTUP
# ================================

if __name__ == "__main__":

    logger.info("🚀 Starting FastAPI server...")
    logger.info("📖 API documentation available at: http://localhost:8080/docs")
    logger.info("🏠 Home page: http://localhost:8080/")
    logger.info("⏹️ Stop server: Ctrl+C")

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8080,
        log_level="info",
        reload=True
    )