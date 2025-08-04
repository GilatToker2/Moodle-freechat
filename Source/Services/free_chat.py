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
        logger.info(f"RAG System initialized with index: {index_name}, model: {self.chat_model}")


    async def generate_answer(
            self,
            conversation_id: str,
            conversation_history: List[Dict],
            course_id: str,
            user_message: str,
            stage: str,
            source_id: str = None,
            top_k: int = 5,
            max_tokens: int = 10000,
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
            max_tokens: Maximum tokens for response
            temperature: Creativity level (0-1)

        Returns:
            Dict with all required fields, final answer, and sources
        """
        try:
            logger.debug(f"Processing RAG query: {user_message}")

            # Build context from conversation history
            conversation_context = ""
            if conversation_history:
                conversation_context = "\nPrevious conversation:\n"
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    conversation_context += f"{role}: {content}\n"

            # Step 1: Search relevant chunks using semantic search
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

            # Step 2: Build context from chunks
            context = self._build_context_from_chunks(search_results)

            # Step 3: Build prompt with conversation context
            prompt = self._build_rag_prompt(user_message, context, conversation_context)

            # Step 4: Send to language model
            response = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            final_answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated answer for query: {user_message}")

            # Step 5: Process sources - always return sources
            sources = self._extract_sources_info(search_results)

            # Return complete structure with all required fields
            return {
                "conversation_id": conversation_id,
                "conversation_history": conversation_history,
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

    def _build_rag_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """
        Build structured prompt for language model including conversation history
        """
        prompt = f"""Based on the following information, please answer the question accurately and in detail:

{conversation_context}

Relevant context from knowledge base:
{context}

Current question: {query}

Guidelines for response:
1. Answer in Hebrew clearly and understandably
2. Base your answer only on the information provided in the context
3. Consider the conversation history for context
4. If the information is not sufficient for a complete answer, mention this
5. Organize the answer logically and clearly

Answer:"""

        return prompt

    def _get_system_prompt(self) -> str:
        """
        Define system behavior
        """
        return """You are an expert AI assistant that answers questions based on provided information.

Your role:
- Answer in Hebrew accurately and helpfully
- Base answers only on the information provided in the context and do not invent information that doesn't exist in the sources
- Organize answers clearly and understandably
- Mention if information is insufficient for a complete answer

Response style:
- Respond like an encouraging and interactive chatbot, not just a static answer
- Use a friendly and pedagogical tone to support learning and exploration

- Clear and professional
- Structured and organized
- Suitable for students and learners
- Include examples when relevant"""

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
                "text_preview": chunk.get('text', '')[:150] + "..." if len(chunk.get('text', '')) > 150 else chunk.get('text', '')
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
        print("🚀 Free Chat System Demo")
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
            user_message="מה זה טבלת אמת ואי ניתן לבנות אותה?",
            stage="regular_chat"
        )

        print(f"✅ Conversation ID: {result['conversation_id']}")
        print(f"📚 Course ID: {result['course_id']}")
        print(f"🎭 Stage: {result['stage']}")
        print(f"💬 User Message: {result['user_message']}")
        print(f"🤖 Final Answer: {result['final_answer']}")
        print(f"📖 Sources ({len(result.get('sources', []))}):")
        for i, source in enumerate(result.get('sources', []), 1):
            print(f"  Source {i}:")
            print(f"    📄 Source ID: {source.get('source_id', 'N/A')}")
            print(f"    📚 Course ID: {source.get('course_id', 'N/A')}")
            print(f"    📑 Chunk: {source.get('chunk_index', 'N/A')}")
            print(f"    ⭐ Score: {source.get('relevance_score', 0):.3f}")
            if source.get('start_time'):
                print(f"    ⏰ Time: {source.get('start_time', '')} - {source.get('end_time', '')}")
            if source.get('section_title'):
                print(f"    📋 Section: {source.get('section_title', '')}")
            print(f"    📜 Preview: {source.get('text_preview', '')}")
            print()
        print(f"⏰ Timestamp: {result['timestamp']}")
        print(f"✅ Success: {result['success']}")
        if not result['success']:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")


    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
