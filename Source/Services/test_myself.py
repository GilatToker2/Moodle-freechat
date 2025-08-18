"""
Assistant Helper - Simple AI assistance for course content
"""

import logging
import asyncio
from typing import Dict, Optional, List
from openai import AsyncAzureOpenAI
from datetime import datetime

from Source.Services.search_on_index import AdvancedUnifiedContentSearch
from Source.Services.prompt_loader import get_prompt_loader
from Config.config import (
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_COMPLETION_MODEL, INDEX_NAME
)
from Config.logging_config import setup_logging

logger = setup_logging()


class AssistantHelper:
    """Simple AI Assistant for course content"""

    def __init__(self,
                 openai_client: AsyncAzureOpenAI = None,
                 search_system: AdvancedUnifiedContentSearch = None,
                 prompt_loader = None):
        """
        Initialize Assistant Helper

        Args:
            openai_client: Shared OpenAI client
            search_system: Shared search system
            prompt_loader: Shared prompt loader
        """
        # Use provided objects or create fallbacks
        self.search_system = search_system or AdvancedUnifiedContentSearch(INDEX_NAME)
        self.openai_client = openai_client or AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        self.prompt_loader = prompt_loader or get_prompt_loader()
        self.chat_model = AZURE_OPENAI_CHAT_COMPLETION_MODEL

    async def get_help(
            self,
            conversation_id: str,
            conversation_history: List[Dict],
            mode: str,  # "lecture" or "full_course"
            identifier: str,  # course_id or source_id
            query: str,  # user question
    ) -> Dict:
        """
        Get AI assistance with conversation context

        Args:
            conversation_id: unique conversation identifier
            conversation_history: list of previous messages
            mode: "lecture" (specific file) or "full_course" (entire course)
            identifier: course_id (for full_course) or source_id (for lecture)
            query: user question

        Returns:
            Dict with response and sources
        """
        try:
            # Search for relevant content based on mode
            if mode == "lecture":
                # Search in specific source/file
                results = await self.search_system.semantic_search(
                    query=query,
                    top_k=5,
                    source_id=identifier
                )
            elif mode == "full_course":
                # Search in entire course
                results = await self.search_system.semantic_search(
                    query=query,
                    top_k=8,
                    course_id=identifier
                )
            else:
                return {
                    "conversation_id": conversation_id,
                    "mode": mode,
                    "identifier": identifier,
                    "query": query,
                    "response": f"מצב לא תקין: {mode}. השתמש ב-'lecture' או 'full_course'",
                    "sources": [],
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }

            if not results:
                return {
                    "conversation_id": conversation_id,
                    "mode": mode,
                    "identifier": identifier,
                    "query": query,
                    "response": "לא נמצא תוכן רלוונטי לשאלתך",
                    "sources": [],
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                }

            # Build context from results
            context = self._build_context(results)

            # Build conversation context
            conversation_context = self._build_conversation_context(conversation_history)

            # Get prompts using injected prompt_loader
            system_prompt = self.prompt_loader.get_prompt("test_myself", "system")
            user_prompt = self.prompt_loader.get_prompt(
                "test_myself",
                "user",
                conversation_context=conversation_context,
                context=context,
                query=query
            )

            # Get AI response
            response = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=8000,
                temperature=0.4
            )

            ai_response = response.choices[0].message.content.strip()

            # Format sources
            sources = []
            for i, result in enumerate(results, 1):
                sources.append({
                    "index": i,
                    "source_id": result.get('source_id', ''),
                    "content_type": result.get('content_type', ''),
                    "score": result.get('@search.score', 0),
                    "preview": result.get('text', '')[:150] + "..."
                })

            # Build updated conversation history with timestamps
            updated_conversation_history = conversation_history.copy() if conversation_history else []

            # Add user query to history
            updated_conversation_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })

            # Add assistant response to history
            updated_conversation_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "conversation_id": conversation_id,
                "conversation_history": updated_conversation_history,
                "mode": mode,
                "identifier": identifier,
                "query": query,
                "response": ai_response,
                "sources": sources,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in assistant helper: {e}")
            return {
                "conversation_id": conversation_id,
                "mode": mode,
                "identifier": identifier,
                "query": query,
                "response": f"שגיאה: {str(e)}",
                "sources": [],
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _build_context(self, results):
        """Build simple context from search results"""
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result.get('text', '')
            context_parts.append(f"מקור {i}: {text}")
        return "\n\n".join(context_parts)

    def _build_conversation_context(self, conversation_history):
        """Build conversation context from history"""
        if not conversation_history:
            return "זוהי תחילת השיחה."

        context_parts = ["הקשר השיחה הקודמת:"]
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = "סטודנט" if msg.get('role') == 'user' else "מורה"
            content = msg.get('content', '')
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)


async def main():
    """Demo"""
    assistant = AssistantHelper()

    # Test lecture mode
    result1 = await assistant.get_help(
        conversation_id="demo-123",
        conversation_history=[],
        mode="lecture",
        identifier="13",  # source_id
        query="מה זה יחס שקילות?"
    )

    print("=== Lecture Mode Test ===")
    print(f"Response: {result1['response']}")
    print(f"Sources: {len(result1['sources'])}")

    # Test full course mode
    result2 = await assistant.get_help(
        conversation_id="demo-456",
        conversation_history=[],
        mode="full_course",
        identifier="Discrete_mathematics",  # course_id
        query="איך אני יכול להכין למבחן?"
    )

    print("\n=== Full Course Mode Test ===")
    print(f"Response: {result2['response']}")
    print(f"Sources: {len(result2['sources'])}")


if __name__ == "__main__":
    asyncio.run(main())
