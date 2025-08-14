"""
Academic Content Processing API

API Documentation: http://localhost:8080/docs

Required old_config.py settings:
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
from contextlib import asynccontextmanager
from Config.logging_config import setup_logging
from Source.Services.free_chat import RAGSystem
from Source.Services.test_myself import AssistantHelper

# Initialize logger
logger = setup_logging()


# Convenience function for backward compatibility
def debug_log(message):
    """Write debug message using proper logging"""
    logger.debug(message)


# Global variables to hold systems
rag_system = None
assistant_helper = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan Context Manager - Application lifecycle management

    This function manages the application startup and shutdown process:
    - Startup: Initialize connections and services
    - Shutdown: Close connections gracefully

    Managed connections:
    - Azure OpenAI clients (AsyncAzureOpenAI)
    - Azure Search clients (SearchClient)
    - Background tasks and async workers
    """
    global rag_system, assistant_helper

    # STARTUP - Application initialization
    logger.info("App is starting...")
    logger.info("Initializing connections...")

    try:
        # Initialize RAG system with all its connections
        rag_system = RAGSystem()
        logger.info("RAG System initialized successfully")

        # Initialize Assistant Helper
        assistant_helper = AssistantHelper()
        logger.info("Assistant Helper initialized successfully")

        logger.info("Azure OpenAI client connected")
        logger.info("Azure Search client connected")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    # Application runs here...
    yield

    # SHUTDOWN - Application cleanup
    logger.info("App is shutting down...")
    logger.info("Closing connections gracefully...")

    try:
        if rag_system:
            # Close RAG System OpenAI client
            if hasattr(rag_system, 'openai_client') and rag_system.openai_client:
                await rag_system.openai_client.close()
                logger.info("RAG System OpenAI client closed")

            # Close Search System resources
            if hasattr(rag_system, 'search_system') and rag_system.search_system:
                # Close search system OpenAI client
                if hasattr(rag_system.search_system, 'openai_client') and rag_system.search_system.openai_client:
                    await rag_system.search_system.openai_client.close()
                    logger.info("Search system OpenAI client closed")

                # Close Azure Search client (if it has async close method)
                search_client = getattr(rag_system.search_system, 'search_client', None)
                if search_client and hasattr(search_client, 'close'):
                    search_client.close()
                    logger.info("Azure Search client closed")

            # Close Blob Manager resources
            if hasattr(rag_system, 'blob_manager') and rag_system.blob_manager:
                if hasattr(rag_system.blob_manager, '_async_client') and rag_system.blob_manager._async_client:
                    await rag_system.blob_manager._async_client.close()
                    logger.info("Blob manager client closed")

        # Close Assistant Helper resources
        if assistant_helper:
            if hasattr(assistant_helper, 'openai_client') and assistant_helper.openai_client:
                await assistant_helper.openai_client.close()
                logger.info("Assistant Helper OpenAI client closed")

        logger.info("All connections closed successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Chat Service API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# RESPONSE MODELS
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
    conversation_history: List[Dict[str, Any]]
    course_id: str
    user_message: str
    stage: str
    final_answer: str
    sources: List[Dict[str, Any]]
    timestamp: str
    success: bool


class AssistantRequest(BaseModel):
    conversation_id: str
    conversation_history: List[Dict[str, Any]]
    mode: str  # "lecture" or "full_course"
    identifier: str  # course_id or source_id
    query: str  # user question


class AssistantResponse(BaseModel):
    conversation_id: str
    mode: str
    identifier: str
    query: str
    response: str
    sources: List[Dict[str, Any]]
    success: bool
    timestamp: str


# ================================
# ROOT & HEALTH ENDPOINTS
# ================================

@app.get("/", tags=["System"])
async def root():
    """Home page - Search & Chat Service information"""
    return {
        "message": "Search & Chat Service",
        "version": "1.0.0",
        "status": "Active",
        "functions": [
            "/free-chat - RAG-based conversational AI",
            "/test_myself - AI tutor for self-assessment and guided learning",
            "/search - Advanced content search",
            "/index/status - Check index status"
        ],
        "docs_url": "/docs"
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "search-chat-service"}


# ================================
# FREE CHAT ENDPOINTS
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
        Free Chat with RAG-based Responses

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
            {
              "role": "user",
              "content": "User query: מה זה לוגיקה?\\n\\nRelevant context:\\nSource 1: לוגיקה היא תחום במתמטיקה העוסק בחוקי החשיבה הנכונה",
              "timestamp": "2025-01-14T08:45:00.123456"
            },
            {
              "role": "assistant",
              "content": "לוגיקה היא תחום יסודי במתמטיקה שעוסק בחוקי החשיבה הנכונה",
              "timestamp": "2025-01-14T08:45:02.789012"
            }
          ],
          "course_id": "Discrete_mathematics",
          "user_message": "תן לי דוגמא לטבלת אמת",
          "stage": "regular_chat"
        }
        ```


    **Parameters:**
    - **conversation_id**: Unique identifier for the conversation
    - **conversation_history**: List of previous messages for context (each message includes role, content, and timestamp)
    - **course_id**: Course identifier to filter relevant content
    - **user_message**: Current user question/message
    - **stage**: User stage (regular_chat/quiz_mode/presentation_discussion)
    - **source_id**: Optional - filter to specific source (video/document)

    **Returns:**
    - All input fields preserved
    - **conversation_history**: Updated conversation history including new exchange with context chunks
    - **final_answer**: RAG-based response in Hebrew
    - **sources**: Detailed information about sources used
    - **timestamp**: Response generation time
    - **success**: Boolean indicating operation success
    """
    try:
        logger.info(f"Free chat request: {request.user_message} (course: {request.course_id})")

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
        result = await rag_system.generate_answer(
            conversation_id=request.conversation_id,
            conversation_history=request.conversation_history,
            course_id=request.course_id,
            user_message=request.user_message,
            stage=request.stage,
            source_id=request.source_id
        )

        # Log result
        if result['success']:
            logger.info(f"Generated answer for: {request.user_message}")
        else:
            logger.warning(f"Failed to generate answer: {result.get('error', 'Unknown error')}")

        # Return complete response with updated conversation_history
        return FreeChatResponse(
            conversation_id=result['conversation_id'],
            conversation_history=result['conversation_history'],
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
        logger.error(f"Error in free chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Free chat failed: {str(e)}")


# ================================
# ASSISTANT HELPER ENDPOINTS
# ================================

@app.post(
    "/test_myself",
    response_model=AssistantResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Test myself failed"}
    },
    tags=["Test Myself"]
)
async def test_myself_endpoint(request: AssistantRequest):
    """
    Test Myself - AI Tutor for Self-Assessment and Learning

    **Function Description:**
    A specialized chatbot designed to help students learn through self-testing and guided discovery.
    Instead of providing direct answers, it generates strategic questions that lead students to understand concepts deeply.

    **What to Expect:**
    • Generates targeted questions to test student understanding
    • Creates step-by-step learning paths through questioning
    • Helps students discover knowledge gaps and fill them
    • Provides hints and guidance without giving away answers
    • Encourages active learning and critical thinking
    • Based only on indexed course content
    • Supports both lecture-specific and full-course learning

    **Request Body Example:**
    ```json
    {
        "conversation_id": "demo-123",
        "conversation_history": [
            {"role": "user", "content": "היי", "timestamp": "2025-01-14T08:45:00.123456"},
            {"role": "assistant", "content": "שלום, איך אוכל לעזור לך היום?", "timestamp": "2025-01-14T08:45:01.789012"}
        ],
        "mode": "lecture",
        "identifier": "13",
        "query": "אני רוצה ללמוד מה זה יחס שקילות?"
    }
    ```

    **Parameters:**
    - **conversation_id**: Unique identifier for the conversation session
    - **conversation_history**: List of previous messages in the conversation for context
        - Each message contains: role ("user"/"assistant"), content (message text), and timestamp
    - **mode**: "lecture" (specific file/source) or "full_course" (entire course)
    - **identifier**: source_id (for lecture mode) or course_id (for full_course mode)
    - **query**: Student's current question or request for help

    **Returns:**
    - **mode**: The assistance mode used
    - **identifier**: The identifier used for search
    - **query**: The original student query
    - **response**: Educational AI response with guiding questions and hints
    - **sources**: Information about content sources used
    - **success**: Boolean indicating operation success
    - **timestamp**: Response generation time
    """
    try:
        logger.info(f"Assistant help request: {request.query} (mode: {request.mode}, id: {request.identifier})")

        # Validate required fields
        if not request.conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id is required")
        if not request.mode:
            raise HTTPException(status_code=400, detail="mode is required")
        if not request.identifier:
            raise HTTPException(status_code=400, detail="identifier is required")
        if not request.query:
            raise HTTPException(status_code=400, detail="query is required")

        # Validate mode
        if request.mode not in ["lecture", "full_course"]:
            raise HTTPException(status_code=400, detail="mode must be 'lecture' or 'full_course'")

        # Call Assistant Helper
        result = await assistant_helper.get_help(
            conversation_id=request.conversation_id,
            conversation_history=request.conversation_history,
            mode=request.mode,
            identifier=request.identifier,
            query=request.query
        )

        # Log result
        if result['success']:
            logger.info(f"Generated assistant help for: {request.query}")
        else:
            logger.warning(f"Failed to generate assistant help: {result.get('error', 'Unknown error')}")

        # Return response
        return AssistantResponse(
            conversation_id=result['conversation_id'],
            mode=result['mode'],
            identifier=result['identifier'],
            query=result['query'],
            response=result['response'],
            sources=result['sources'],
            success=result['success'],
            timestamp=result['timestamp']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in assistant help endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Assistant help failed: {str(e)}")


# ================================
# SERVER STARTUP
# ================================

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    logger.info("API documentation available at: http://localhost:8080/docs")
    logger.info("Home page: http://localhost:8080/")
    logger.info("Stop server: Ctrl+C")

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8080,
        log_level="info",
        reload=True
    )
