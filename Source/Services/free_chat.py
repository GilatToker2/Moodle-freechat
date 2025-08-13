"""
Free Chat System - Conversation Management with RAG
System for managing conversations with user state tracking and content

Process:
1. Receive user query with conversation context
2. Search relevant chunks in index (using search_on_index)
3. Build prompt with relevant context
4. Send to language model
5. Return response with full conversation data

Query Response Fields:
- conversation_id
- conversation_history
- course_id
- user_message
- stage (user state: quiz_mode/regular_chat/presentation_discussion)
"""

import logging
import asyncio
from typing import List, Dict, Optional
from openai import AsyncAzureOpenAI
from datetime import datetime

from Source.Services.search_on_index import AdvancedUnifiedContentSearch
from Source.Services.blob_manager import BlobManager
from Config.config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL, INDEX_NAME
)
from Config.logging_config import setup_logging

# Initialize logger
logger = setup_logging()


class RAGSystem:
    """
    Complete RAG System - Search + Answer Generation
    """

    def __init__(self, index_name: str = INDEX_NAME):
        """
        Initialize RAG System

        Args:
            index_name: Index name for search
        """
        self.index_name = index_name
        self.search_system = AdvancedUnifiedContentSearch(index_name)

        self.openai_client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        self.chat_model = AZURE_OPENAI_CHAT_COMPLETION_MODEL
        self.blob_manager = BlobManager()
        logger.info(f"RAG System initialized with index: {index_name}, model: {self.chat_model}")

    async def close(self):
        """Close all async resources"""
        try:
            # Close OpenAI client
            if hasattr(self, 'openai_client') and self.openai_client:
                await self.openai_client.close()
                logger.info("RAG System OpenAI client closed")

            # Close search system resources
            if hasattr(self, 'search_system') and self.search_system:
                if hasattr(self.search_system, 'openai_client') and self.search_system.openai_client:
                    await self.search_system.openai_client.close()
                    logger.info("Search system OpenAI client closed")

            # Close blob manager resources
            if hasattr(self, 'blob_manager') and self.blob_manager:
                if hasattr(self.blob_manager, '_async_client') and self.blob_manager._async_client:
                    await self.blob_manager._async_client.close()
                    logger.info("Blob manager client - closed")

        except Exception as e:
            logger.error(f"Error closing RAG System resources: {e}")

    async def _load_syllabus(self, course_id: str) -> str:
        """
        Load syllabus content for the given course_id from blob storage

        Args:
            course_id: Course identifier

        Returns:
            Syllabus content as string, or empty string if not found
        """
        try:
            syllabus_blob_name = f"{course_id}/syllabus.md"

            # Download syllabus content to memory
            syllabus_bytes = await self.blob_manager.download_to_memory(syllabus_blob_name)

            if syllabus_bytes:
                syllabus_content = syllabus_bytes.decode('utf-8')
                logger.info(f"Loaded syllabus for course {course_id} from blob: {syllabus_blob_name}")
                return syllabus_content
            else:
                logger.warning(f"Syllabus file not found in blob: {syllabus_blob_name}")
                return ""

        except Exception as e:
            logger.error(f"Error loading syllabus for course {course_id}: {e}")
            return ""

    async def generate_answer(
            self,
            conversation_id: str,
            conversation_history: List[Dict],
            course_id: str,
            user_message: str,
            stage: str,
            source_id: str = None,
            top_k: int = 5,
            temperature: float = 0.3
    ) -> Dict:
        """
        Main function - Generate RAG-based answer with all conversation fields

        Args:
            conversation_id: conversation identifier
            conversation_history: JSON list of previous messages
            course_id: course identifier
            user_message: current user message
            stage: user stage (regular_chat/quiz_mode/presentation_discussion)
            source_id: optional source filter
            top_k: Number of chunks to retrieve from search
            temperature: Creativity level (0-1)

        Returns:
            Dict with all required fields, final answer, and sources
        """
        try:
            logger.debug(f"Processing RAG query: {user_message}")

            # Step 1: Load syllabus for the course
            syllabus_content = await self._load_syllabus(course_id)

            # Step 2: Search relevant chunks using semantic search
            search_results = await self.search_system.semantic_search(
                query=user_message,
                top_k=top_k,
                source_id=source_id,
                course_id=course_id
            )

            if not search_results:
                logger.warning(f"No relevant content found for query: {user_message}")
                return {
                    "conversation_id": conversation_id,
                    "conversation_history": conversation_history,
                    "course_id": course_id,
                    "user_message": user_message,
                    "stage": stage,
                    "final_answer": "מצטער, לא מצאתי מידע רלוונטי לשאלתך במאגר הידע. אנא נסה לנסח את השאלה בצורה אחרת.",
                    "sources": [],
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": "No relevant content found in RAG"
                }

            logger.debug(f"Found {len(search_results)} relevant chunks")

            # Step 3: Build context from chunks
            context = self._build_context_from_chunks(search_results)

            # Step 4: Build messages array with conversation history and current query
            messages = self._build_conversation_messages(conversation_history, user_message, context, syllabus_content)

            # Log the final prompt for debugging
            logger.info("=== Final prompt being sent to model ===")
            for i, message in enumerate(messages):
                logger.info(f"Message {i+1} - Role: {message['role']}")
                logger.info(f"Content:\n{message['content']}")
                logger.info("=" * 50)

            # Step 4: Send to language model
            response = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature
            )

            final_answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated answer for query: {user_message}")

            # Step 5: Process sources - always return sources
            sources = self._extract_sources_info(search_results)

            # Step 6: Build updated conversation history including the new exchange
            updated_conversation_history = conversation_history.copy() if conversation_history else []

            # Add the current user message WITH context (as it was sent to the model)
            user_message_with_context = f"""User query: {user_message}

Relevant context:
{context}"""

            updated_conversation_history.append({
                "role": "user",
                "content": user_message_with_context,
                "timestamp": datetime.now().isoformat()
            })

            # Add the assistant response to history
            updated_conversation_history.append({
                "role": "assistant",
                "content": final_answer,
                "timestamp": datetime.now().isoformat()
            })

            # Return complete structure with updated conversation history
            return {
                "conversation_id": conversation_id,
                "conversation_history": updated_conversation_history,
                "course_id": course_id,
                "user_message": user_message,
                "stage": stage,
                "final_answer": final_answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            return {
                "conversation_id": conversation_id,
                "conversation_history": conversation_history,
                "course_id": course_id,
                "user_message": user_message,
                "stage": stage,
                "final_answer": f"Error: {str(e)}",
                "sources": [],
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            }

    def _build_context_from_chunks(self, chunks: List[Dict]) -> str:
        """
        Build focused context from found chunks
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', '')

            # Simple source numbering - don't worry about metadata fields
            source_info = f"Source {i}"
            context_parts.append(f"{source_info}:\n{text}")

        return "\n\n".join(context_parts)

    def _build_conversation_messages(self, conversation_history: List[Dict], user_message: str, context: str, syllabus_content: str = "") -> List[
        Dict]:
        """
        Build messages array with proper conversation structure including system prompt,
        conversation history, and current query with context and syllabus
        """
        messages = []

        # Add system message with syllabus context
        messages.append({
            "role": "system",
            "content": self._get_system_prompt(syllabus_content)
        })

        # Add conversation history (all messages)
        if conversation_history:
            for msg in conversation_history:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                # Ensure role is valid (assistant or user)
                if role not in ['assistant', 'user']:
                    role = 'user'

                messages.append({
                    "role": role,
                    "content": content
                })

        # Add current user message with context
        user_message_with_context = f"""User query: {user_message}

        Relevant context:
        {context}"""

        messages.append({
            "role": "user",
            "content": user_message_with_context
        })

        return messages

    def _get_system_prompt(self, syllabus_content: str = "") -> str:
        """
        Define system behavior with optional syllabus context
        """
        base_prompt = """You are an expert Hebrew-speaking tutoring assistant that operates inside a RAG pipeline.

        Your role:
        - Answer in Hebrew accurately and helpfully
        - Base your answer only on the information provided in the context
        - Consider the conversation history for context
        - If the information is not sufficient for a complete answer, mention this
        - Organize the answer logically and clearly
        - This is a dedicated model to help students learn the material. If asked about something unrelated, respond that this is not its role.

        Response style:
        - Respond like an encouraging and interactive chatbot, not just a static answer
        - Use a friendly and pedagogical tone to support learning and exploration
        - Clear and professional
        - Structured and organized
        - Suitable for students and learners
        - Include examples when relevant

        Guidelines for using the context:
        1. Answer based on the information provided in the context above
        2. Consider the conversation history for better understanding
        3. If the context doesn't contain sufficient information for a complete answer, mention this"""

        # Add syllabus context if available
        if syllabus_content:
            syllabus_section = f"""

        COURSE SYLLABUS:
        The following is the course syllabus that provides important context about the course structure, topics, and learning objectives:

        {syllabus_content}

        Use this syllabus information to:
        - Better understand the course context and structure
        - Reference relevant topics from the syllabus when appropriate
        - Help students understand how topics fit into the overall course structure"""

            return base_prompt + syllabus_section

        return base_prompt

    def _extract_sources_info(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract source information for display - simplified without content_type dependencies
        """
        sources = []

        for i, chunk in enumerate(chunks, 1):
            # Extract all available fields without worrying about content_type
            source_info = {
                "index": i,
                "source_id": chunk.get('source_id', ''),
                "course_id": chunk.get('course_id', ''),
                "chunk_index": chunk.get('chunk_index', 0),
                "relevance_score": chunk.get('@search.score', 0),
                "text_preview": chunk.get('text', '')
            }

            # Add any additional fields that exist (don't filter by content_type)
            if chunk.get('start_time'):
                source_info["start_time"] = chunk.get('start_time', '')
            if chunk.get('end_time'):
                source_info["end_time"] = chunk.get('end_time', '')
            if chunk.get('section_title'):
                source_info["section_title"] = chunk.get('section_title', '')

            sources.append(source_info)

        return sources


async def main():
    """
    Main function - Free Chat Demo
    """
    try:
        logger.info("Starting Free Chat System Demo")
        print("Free Chat System Demo")
        print("=" * 50)

        # Initialize RAG system
        rag = RAGSystem()

        # Test generate_answer function directly
        logger.info("Testing generate_answer function")
        result = await rag.generate_answer(
            conversation_id="demo-123",
            conversation_history=[
                {"role": "user", "content": "Hello", "timestamp": "2025-01-01T10:00:00"},
                {"role": "assistant", "content": "Hi there!", "timestamp": "2025-01-01T10:00:01"}
            ],
            course_id="Discrete_mathematics",
            user_message="מה זה טבלת אמת ",
            stage="regular_chat"
        )

        print(f"Conversation ID: {result['conversation_id']}")
        print(f"Course ID: {result['course_id']}")
        print(f"Stage: {result['stage']}")
        print(f"User Message: {result['user_message']}")
        print(f"Final Answer: {result['final_answer']}")

        print(f"\nUpdated Conversation History ({len(result.get('conversation_history', []))}) messages:")
        for i, msg in enumerate(result.get('conversation_history', []), 1):
            print(f"  Message {i} - Role: {msg.get('role', 'unknown')}")
            print(f"    Timestamp: {msg.get('timestamp', 'N/A')}")
            content = msg.get('content', '')
            print(f"    Content: {content}")
            print()

        print(f"Sources ({len(result.get('sources', []))}):")
        for i, source in enumerate(result.get('sources', []), 1):
            print(f"  Source {i}:")
            print(f"    Source ID: {source.get('source_id', 'N/A')}")
            print(f"    Course ID: {source.get('course_id', 'N/A')}")
            print(f"    Chunk: {source.get('chunk_index', 'N/A')}")
            print(f"    Score: {source.get('relevance_score', 0):.3f}")
            if source.get('start_time'):
                print(f"    Time: {source.get('start_time', '')} - {source.get('end_time', '')}")
            if source.get('section_title'):
                print(f"    Section: {source.get('section_title', '')}")
            print(f"    Preview: {source.get('text_preview', '')}")
            print()
        print(f"Timestamp: {result['timestamp']}")
        print(f"Success: {result['success']}")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")

        # Close RAG system resources
        await rag.close()
        logger.info("RAG system resources closed successfully")

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
