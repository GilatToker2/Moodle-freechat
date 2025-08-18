"""
Advanced Unified Content Search - Advanced search system for unified content
Supports searching in videos and documents
"""
import logging
import asyncio
from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncAzureOpenAI
import traceback
from Config.config import (
    SEARCH_SERVICE_NAME, SEARCH_API_KEY,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_MODEL, INDEX_NAME
)
from Config.logging_config import setup_logging

# Initialize logger
logger = setup_logging()


class AdvancedUnifiedContentSearch:
    """
    Advanced search system for unified content - videos and documents
    Supports textual, semantic and vector search
    Allows searching all content together or filtered by type
    """

    def __init__(self, index_name: str = INDEX_NAME):
        self.index_name = INDEX_NAME
        self.search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
        self.credential = AzureKeyCredential(SEARCH_API_KEY)

        # Create search client
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

        # Create OpenAI client for vector search
        self.openai_client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

        logger.info(f"AdvancedUnifiedContentSearch initialized with index: {self.index_name}")

    def check_index_status(self) -> Dict:
        """Check unified index status and display basic information"""
        logger.info("=" * 60)

        try:
            # General search for testing
            results = self.search_client.search(
                search_text="*",
                select=["*"],
                top=5,
                include_total_count=True
            )

            total_count = results.get_count()
            docs = list(results)

            logger.info(f"Total chunks in unified index: {total_count}")
            logger.info(f"Documents returned for testing: {len(docs)}")

            if docs:
                logger.info(f"Unified index is active and contains data")

                # Count by content type
                video_results = self.search_client.search("*", filter="content_type eq 'video'",
                                                          include_total_count=True, top=0)
                video_count = video_results.get_count()

                doc_results = self.search_client.search("*", filter="content_type eq 'document'",
                                                        include_total_count=True, top=0)
                doc_count = doc_results.get_count()

                logger.info(f"Video chunks: {video_count}")
                logger.info(f"Document chunks: {doc_count}")

                # Display document examples
                logger.info(f"\nDocument examples in index:")
                for i, doc in enumerate(docs[:10], 1):
                    content_type = doc.get('content_type', 'unknown')
                    logger.info(f"\nDocument {i} ({content_type}):")
                    logger.info(f"  ID: {doc.get('id', 'N/A')}")
                    logger.info(f"  Source ID: {doc.get('source_id', 'N/A')}")
                    logger.info(f"  Source Name: {doc.get('source_name', 'N/A')}")
                    logger.info(f"  Chunk Index: {doc.get('chunk_index', 'N/A')}")

                    if content_type == 'video':
                        logger.info(f"  Start Time: {doc.get('start_time', 'N/A')}")
                        logger.info(f"  Start Seconds: {doc.get('start_seconds', 'N/A')}")
                    elif content_type == 'document':
                        logger.info(f"  Section Title: {doc.get('section_title', 'N/A')}")
                        logger.info(f"  Document Type: {doc.get('document_type', 'N/A')}")

                    # Display text content
                    text = doc.get('text', '')
                    if text:
                        preview = text[:150] + "..." if len(text) > 150 else text
                        logger.info(f"  Content: {preview}")
                    logger.info("-" * 30)

                return {
                    "status": "active",
                    "total_chunks": total_count,
                    "video_chunks": video_count,
                    "document_chunks": doc_count,
                    "sample_doc": docs[0] if docs else None
                }
            else:
                logger.info("Index exists but is empty")
                return {"status": "empty", "total_chunks": 0}

        except Exception as e:
            logger.info(f"Error accessing index: {e}")
            logger.error(f"Error checking index status: {e}")
            return {"status": "error", "error": str(e)}

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        try:
            response = await self.openai_client.embeddings.create(
                model=AZURE_OPENAI_EMBEDDING_MODEL,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

    async def simple_text_search(self, query: str, top_k: int = 5, source_id: str = None, course_id: str = None) -> List[Dict]:
        """Simple text search when embedding cannot be extracted"""
        logger.info("=" * 60)

        try:
            search_params = {
                "search_text": query,
                "select": [
                    "id", "content_type", "source_id", "course_id", "chunk_index",
                    "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics", "file_name"
                ],
                "top": top_k,
                "include_total_count": True
            }

            # Add filters
            filters = []
            if source_id:
                escaped_source_id = source_id.replace("'", "''")
                filters.append(f"source_id eq '{escaped_source_id}'")
            if course_id:
                escaped_course_id = course_id.replace("'", "''")
                filters.append(f"course_id eq '{escaped_course_id}'")

            if filters:
                search_params["filter"] = " and ".join(filters)

            results = self.search_client.search(**search_params)

            docs = list(results)
            total_count = results.get_count()

            if not docs:
                logger.info("No results found")
                return []

            filter_msg = self._build_filter_message(source_id, course_id)
            logger.info(f"Found {len(docs)} results out of {total_count} chunks{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                logger.info(f"\nResult {i} ({content_type_doc}, score: {score:.3f}):")
                logger.info(f"  ID: {doc.get('id', 'N/A')}")
                logger.info(f"  Source ID: {doc.get('source_id', 'N/A')}")
                logger.info(f"  Course ID: {doc.get('course_id', 'N/A')}")
                logger.info(f"  Chunk: {doc.get('chunk_index', 'N/A')}")
                logger.info(f"  Created: {doc.get('created_date', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    end_time = doc.get('end_time', '')
                    if start_time:
                        logger.info(f"  Time: {start_time} - {end_time}")
                    keywords = doc.get('keywords', '')
                    if keywords:
                        logger.info(f"  Keywords: {keywords}")
                    topics = doc.get('topics', '')
                    if topics:
                        logger.info(f"  Topics: {topics}")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        logger.info(f"  Section Title: {section_title}")

                # Display file name if available
                file_name = doc.get('file_name', '')
                if file_name:
                    logger.info(f"  File Name: {file_name}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  Content: {preview}")

                logger.info("—" * 40)

            return docs

        except Exception as e:
            logger.info(f"Error in text search: {e}")
            logger.error(f"Error in text search: {e}")
            return []

    async def hybrid_search(self, query: str, top_k: int = 5, source_id: str = None, course_id: str = None) -> List[Dict]:
        """Hybrid search - combines text and vector"""
        logger.info("=" * 60)

        try:
            # Generate embedding for query
            query_vector = await self.generate_query_embedding(query)
            if not query_vector:
                logger.info("Cannot generate embedding, performing text search only")
                return await self.simple_text_search(query, top_k, source_id, course_id)

            search_params = {
                "search_text": query,
                "vector_queries": [VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=50,
                    fields="vector"
                )],
                "select": [
                    "id", "content_type", "source_id", "course_id", "chunk_index",
                    "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics", "file_name"
                ],
                "top": 50,
                "include_total_count": True
            }

            # Add filters
            filters = []
            if source_id:
                escaped_source_id = source_id.replace("'", "''")
                filters.append(f"source_id eq '{escaped_source_id}'")
            if course_id:
                escaped_course_id = course_id.replace("'", "''")
                filters.append(f"course_id eq '{escaped_course_id}'")

            if filters:
                search_params["filter"] = " and ".join(filters)

            results = self.search_client.search(**search_params)

            docs = list(results)
            total_count = results.get_count()

            if not docs:
                logger.info("No hybrid results found")
                return []

            # Slice to requested top_k for display and return
            docs = docs[:top_k]

            filter_msg = self._build_filter_message(source_id, course_id)
            logger.info(f"Found {len(docs)} hybrid results out of {total_count} chunks{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                logger.info(f"\nResult {i} ({content_type_doc}, combined score: {score:.3f}):")
                logger.info(f"  ID: {doc.get('id', 'N/A')}")
                logger.info(f"  Source ID: {doc.get('source_id', 'N/A')}")
                logger.info(f"  Course ID: {doc.get('course_id', 'N/A')}")
                logger.info(f"  Chunk: {doc.get('chunk_index', 'N/A')}")
                logger.info(f"  Created: {doc.get('created_date', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    end_time = doc.get('end_time', '')
                    if start_time:
                        logger.info(f"  Time: {start_time} - {end_time}")
                    keywords = doc.get('keywords', '')
                    if keywords:
                        logger.info(f"  Keywords: {keywords}")
                    topics = doc.get('topics', '')
                    if topics:
                        logger.info(f"  Topics: {topics}")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        logger.info(f"  Section Title: {section_title}")

                # Display file name if available
                file_name = doc.get('file_name', '')
                if file_name:
                    logger.info(f"  File Name: {file_name}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  Content: {preview}")

                logger.info("—" * 40)

            return docs

        except Exception as e:
            logger.info(f"Error in hybrid search: {e}")
            logger.error(f"Error in hybrid search: {e}")
            return []

    async def semantic_search(self, query: str, top_k: int = 5, source_id: str = None, course_id: str = None) -> List[Dict]:
        """Advanced semantic search"""
        logger.info("=" * 60)

        try:
            # Generate embedding for query
            query_vector = await self.generate_query_embedding(query)
            if not query_vector:
                logger.info("Cannot generate embedding, performing text search only")
                return await self.simple_text_search(query, top_k, source_id, course_id)

            # Prepare search parameters
            search_params = {
                "search_text": query,
                "query_type": "semantic",
                "semantic_configuration_name": "default",
                "query_language": "he-il",
                "highlight_fields": "text",
                "vector_queries": [VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="vector"
                )],
                "select": [
                    "id", "content_type", "source_id", "course_id", "chunk_index",
                    "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics", "file_name"
                ],
                "top": top_k
            }

            # Add filters
            filters = []
            if source_id:
                escaped_source_id = source_id.replace("'", "''")
                filters.append(f"source_id eq '{escaped_source_id}'")
            if course_id:
                escaped_course_id = course_id.replace("'", "''")
                filters.append(f"course_id eq '{escaped_course_id}'")

            if filters:
                search_params["filter"] = " and ".join(filters)

            # Advanced semantic search
            results = self.search_client.search(**search_params)

            docs = list(results)

            if not docs:
                logger.info("No semantic results found")
                return []

            filter_msg = self._build_filter_message(source_id, course_id)
            logger.info(f"Found {len(docs)} semantic results{filter_msg}:")

            for i, doc in enumerate(docs, 1):
                score = doc.get('@search.score', 0)
                content_type_doc = doc.get('content_type', 'unknown')
                logger.info(f"\nResult {i} ({content_type_doc}, semantic score: {score:.3f}):")
                logger.info(f"  ID: {doc.get('id', 'N/A')}")
                logger.info(f"  Source ID: {doc.get('source_id', 'N/A')}")
                logger.info(f"  Course ID: {doc.get('course_id', 'N/A')}")
                logger.info(f"  Chunk: {doc.get('chunk_index', 'N/A')}")
                logger.info(f"  Created: {doc.get('created_date', 'N/A')}")

                if content_type_doc == 'video':
                    start_time = doc.get('start_time', '')
                    end_time = doc.get('end_time', '')
                    if start_time:
                        logger.info(f"  Time: {start_time} - {end_time}")
                    keywords = doc.get('keywords', '')
                    if keywords:
                        logger.info(f"  Keywords: {keywords}")
                    topics = doc.get('topics', '')
                    if topics:
                        logger.info(f"  Topics: {topics}")
                elif content_type_doc == 'document':
                    section_title = doc.get('section_title', '')
                    if section_title:
                        logger.info(f"  Section Title: {section_title}")

                # Display file name if available
                file_name = doc.get('file_name', '')
                if file_name:
                    logger.info(f"  File Name: {file_name}")

                text = doc.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    logger.info(f"  Content: {preview}")

                logger.info("—" * 40)

            return docs

        except Exception as e:
            logger.info(f"Error in advanced semantic search: {e}")
            logger.error(f"Error in semantic search: {e}")
            # Fallback to regular hybrid search
            return await self.hybrid_search(query, top_k, source_id, course_id)

    async def get_adjacent_chunks(self, chunk: Dict) -> List[Dict]:
        """
        Get adjacent chunks (before and after) for a given chunk

        Args:
            chunk: The original chunk to find adjacent chunks for

        Returns:
            List of adjacent chunks (before and after the original chunk)
        """
        try:
            source_id = chunk.get('source_id')
            course_id = chunk.get('course_id')
            chunk_index = chunk.get('chunk_index')

            if not all([source_id, course_id, chunk_index is not None]):
                logger.warning(f"Missing required fields for adjacent chunk search: source_id={source_id}, course_id={course_id}, chunk_index={chunk_index}")
                return []

            adjacent_chunks = []

            # Search for chunk before (chunk_index - 1)
            if chunk_index > 0:
                try:
                    escaped_source_id = source_id.replace("'", "''")
                    escaped_course_id = course_id.replace("'", "''")
                    before_filter = f"source_id eq '{escaped_source_id}' and course_id eq '{escaped_course_id}' and chunk_index eq {chunk_index - 1}"
                    before_results = self.search_client.search(
                        search_text="*",
                        filter=before_filter,
                        select=[
                            "id", "content_type", "source_id", "course_id", "chunk_index",
                            "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics", "file_name"
                        ],
                        top=1
                    )
                    before_docs = list(before_results)
                    if before_docs:
                        adjacent_chunks.extend(before_docs)
                        logger.info(f"Found chunk before: {chunk_index - 1}")
                except Exception as e:
                    logger.warning(f"Error searching for chunk before {chunk_index}: {e}")

            # Search for chunk after (chunk_index + 1)
            try:
                escaped_source_id = source_id.replace("'", "''")
                escaped_course_id = course_id.replace("'", "''")
                after_filter = f"source_id eq '{escaped_source_id}' and course_id eq '{escaped_course_id}' and chunk_index eq {chunk_index + 1}"
                after_results = self.search_client.search(
                    search_text="*",
                    filter=after_filter,
                    select=[
                        "id", "content_type", "source_id", "course_id", "chunk_index",
                        "text", "start_time", "end_time", "section_title", "created_date", "keywords", "topics", "file_name"
                    ],
                    top=1
                )
                after_docs = list(after_results)
                if after_docs:
                    adjacent_chunks.extend(after_docs)
                    logger.info(f"Found chunk after: {chunk_index + 1}")
            except Exception as e:
                logger.warning(f"Error searching for chunk after {chunk_index}: {e}")

            logger.debug(f"Found {len(adjacent_chunks)} adjacent chunks for chunk {chunk_index}")
            return adjacent_chunks

        except Exception as e:
            logger.error(f"Error getting adjacent chunks: {e}")
            return []


    async def search_best_answers(self, query: str, k: int = 5, source_id: str = None, course_id: str = None) -> List[Dict]:
        """
        Enhanced function that receives a question and returns K best answers WITH adjacent chunks
        For each of the top K chunks found, also retrieves the chunk before and after it
        Uses semantic search as default, with fallback to hybrid

        Args:
            query: Search query
            k: Number of best results to return
            source_id: Optional - if specified, will search only in this specific source
            course_id: Optional - if specified, will search only in this specific course

        Returns:
            List of chunks including original results and their adjacent chunks
        """
        try:
            # Step 1: Get the top K semantic search results
            logger.info(f"Searching for top {k} chunks for query: {query}")
            try:
                original_results = await self.semantic_search(query, k, source_id, course_id)
            except Exception:
                # fallback to hybrid if semantic fails
                original_results = await self.hybrid_search(query, k, source_id, course_id)

            if not original_results:
                logger.warning("No original results found")
                return []

            # Step 2: For each result, get adjacent chunks
            all_chunks = []
            seen_chunk_ids = set()  # To avoid duplicates

            for i, chunk in enumerate(original_results):
                # Add the original chunk first
                chunk_id = chunk.get('id')
                if chunk_id and chunk_id not in seen_chunk_ids:
                    all_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
                    logger.debug(f"Added original chunk {i+1}: {chunk.get('source_id')}-{chunk.get('chunk_index')}")

                # Get and add adjacent chunks
                adjacent_chunks = await self.get_adjacent_chunks(chunk)
                for adj_chunk in adjacent_chunks:
                    adj_chunk_id = adj_chunk.get('id')
                    if adj_chunk_id and adj_chunk_id not in seen_chunk_ids:
                        all_chunks.append(adj_chunk)
                        seen_chunk_ids.add(adj_chunk_id)
                        logger.debug(f"Added adjacent chunk: {adj_chunk.get('source_id')}-{adj_chunk.get('chunk_index')}")

            logger.info(f"Total chunks retrieved: {len(all_chunks)} (original: {len(original_results)}, with adjacent: {len(all_chunks) - len(original_results)})")
            return all_chunks

        except Exception as e:
            logger.error(f"Error in search_best_answers: {e}")
            # Fallback to original search without context
            try:
                results = await self.semantic_search(query, k, source_id, course_id)
                return results
            except Exception:
                results = await self.hybrid_search(query, k, source_id, course_id)
                return results

    def _build_filter_message(self, source_id: str = None, course_id: str = None) -> str:
        """Build filter message for display"""
        filter_parts = []
        if source_id:
            filter_parts.append(f"מקור: {source_id}")
        if course_id:
            filter_parts.append(f"קורס: {course_id}")

        if filter_parts:
            return f" (מסונן ל-{', '.join(filter_parts)})"
        return ""

async def run_unified_search_demo():
    """Run full demo of unified search system"""
    logger.info("Advanced unified content search system - videos and documents")
    logger.info("=" * 80)

    try:
        # Create search system
        search_system = AdvancedUnifiedContentSearch("unified-content-chunks")

        # Check index status
        logger.info("\nChecking unified index status:")
        status = search_system.check_index_status()

        if status.get("status") != "active":
            logger.info("Index is not active or empty. Please ensure the index is created and contains data.")
            return

        # Example queries (keeping Hebrew as these are test queries)
        demo_queries = [
            "מה זה טרנזטיביות",
            "מתי יש שוויון בין מחלקות שקילות",
            "איך אפשר לשלול ביטוי"
        ]

        logger.info(f"\nRunning demo with {len(demo_queries)} queries:")

        for i, query in enumerate(demo_queries, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Query {i} of {len(demo_queries)}: '{query}'")
            logger.info(f"{'=' * 80}")


            # 1. Search all content with enhanced context (adjacent chunks)
            logger.info(f"\n1. Enhanced search all content (videos + documents) with adjacent chunks:")
            logger.info("-" * 70)
            results = await search_system.search_best_answers(query, k=5)
            logger.info(f"Enhanced search returned {len(results)} total chunks")

            logger.info("\n" + "=" * 80)

            # # 2. Search specific video with enhanced context
            # logger.info(f"\n2. Enhanced search specific video with adjacent chunks:")
            # logger.info("-" * 60)
            # # Assume we have a video with this ID (you'll need to replace with real ID)
            # sample_video_id = "13"
            # logger.info(f"Search in video: {sample_video_id}")
            # results = await search_system.search_best_answers(query, k=5, source_id=sample_video_id)
            # logger.info(f"Enhanced search returned {len(results)} total chunks")
            #
            # logger.info("\n" + "=" * 80)
            #
            # # 3. Search specific course with enhanced context
            # logger.info(f"\n3. Enhanced search specific course with adjacent chunks:")
            # logger.info("-" * 60)
            # sample_course_id = "Discrete_mathematics"
            # logger.info(f"Search in course: {sample_course_id}")
            # results = await search_system.search_best_answers(query, k=5, course_id=sample_course_id)
            # logger.info(f"Enhanced search returned {len(results)} total chunks")


            # Break between queries
            if i < len(demo_queries):
                logger.info("\n" + "Moving to next query..." + "\n")

        logger.info(f"\nDemo completed successfully!")

    except Exception as e:
        logger.info(f"Error running demo: {e}")
        logger.error(f"Error in demo: {e}")
        traceback.print_exc()


async def main():
    """Main function - run search demo"""
    try:
        await run_unified_search_demo()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
