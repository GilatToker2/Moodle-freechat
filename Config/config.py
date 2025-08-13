"""
Configuration file for  Chat Service
Reads configuration from environment variables for security
"""

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
AZURE_OPENAI_CHAT_COMPLETION_MODEL = os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL", "gpt-4.1")

# Azure Search Configuration
SEARCH_SERVICE_NAME = os.getenv("SEARCH_SERVICE_NAME")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "moodle-index-1")

# Azure Storage Configuration
STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "processeddata")  # Default value if not set

# Validation - ensure critical environment variables are set
def validate_config():
    """Validate that all required environment variables are set for Chat Service"""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "SEARCH_SERVICE_NAME",
        "SEARCH_API_KEY",
        "STORAGE_CONNECTION_STRING"
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    print("All required environment variables are set")

# Optional: Run validation when module is imported
# Uncomment the line below if you want automatic validation
# validate_config()
