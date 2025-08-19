"""
Prompt Loader Service - Loads prompts from MD files
Centralized management of all prompts used in the system
"""

import os
from typing import Dict, Optional, Union
from Config.logging_config import setup_logging

logger = setup_logging()


class PromptLoader:
    """
    Service for loading and managing prompts from MD files with true caching
    """

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize prompt loader

        Args:
            prompts_dir: Directory containing prompt MD files
        """
        self.prompts_dir = prompts_dir
        self._prompts_cache: Dict[str, Dict[str, str]] = {}
        self._cache_loaded = False
        logger.info(f"PromptLoader initialized with directory: {prompts_dir}")

    def _load_prompt_file(self, filename: str) -> Dict[str, str]:
        """
        Load and parse a prompt MD file

        Args:
            filename: Name of the prompt file (without .md extension)

        Returns:
            Dictionary with parsed prompt sections
        """
        file_path = os.path.join(self.prompts_dir, f"{filename}.md")

        if not os.path.exists(file_path):
            logger.error(f"Prompt file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return self._parse_prompt_content(content)

        except Exception as e:
            logger.error(f"Error loading prompt file {file_path}: {e}")
            return {}

    def _parse_prompt_content(self, content: str) -> Dict[str, str]:
        """
        Parse prompt MD content into sections - captures whole section body, not just code fences

        Args:
            content: Raw MD content

        Returns:
            Dictionary with parsed sections
        """
        sections = {}
        lines = content.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            # Check for section headers (## Section Name)
            if line.startswith('## ') and not line.startswith('###'):
                # Save previous section if exists
                if current_section and current_content:
                    # Join all content and strip whitespace
                    section_text = '\n'.join(current_content).strip()
                    sections[current_section.lower()] = section_text

                # Start new section
                current_section = line[3:].strip()
                current_content = []
                continue

            # Add content to current section (everything except the header)
            if current_section:
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            section_text = '\n'.join(current_content).strip()
            sections[current_section.lower()] = section_text

        return sections

    def preload_all_prompts(self) -> None:
        """
        Preload all prompts into cache at startup
        """
        # Map prompt types to file names - fixing name mismatches
        prompt_files = {
            'free_chat': 'free_chat_prompt',
            'test_myself': 'quiz_myself_prompt'
        }

        logger.info("Preloading all prompts into cache...")

        for prompt_type, filename in prompt_files.items():
            try:
                prompts = self._load_prompt_file(filename)
                if prompts:
                    self._prompts_cache[prompt_type] = prompts
                    logger.info(f"Preloaded {prompt_type} prompts from {filename}.md")
                else:
                    logger.warning(f"No content loaded for {prompt_type} from {filename}.md")
            except Exception as e:
                logger.error(f"Failed to preload {prompt_type} prompts: {e}")

        self._cache_loaded = True
        logger.info(f"Preloading complete. Cached {len(self._prompts_cache)} prompt types.")

    def get_prompt(self, prompt_type: str, section: str = "system", reload: bool = False, **kwargs) -> str:
        """
        Get prompt text from cache (or load if not cached) and format with variables

        Args:
            prompt_type: Type of prompt (subject_detection, syllabus_generation, etc.)
            section: Section name from MD file (can be complex like "system - מתמטי עם שם מקצוע")
            reload: Force reload from disk if True
            **kwargs: Variables to format into the prompt

        Returns:
            Formatted prompt string
        """
        # Normalize section to lowercase for lookup
        section_key = section.lower()

        # If we have subject_type, try to build the appropriate section name
        if 'subject_type' in kwargs:
            subject_type = kwargs.get('subject_type', '').lower()
            subject_name = kwargs.get('subject_name', '')

            # Try different section name patterns based on subject type and name
            if subject_type == 'מתמטי' and subject_name:
                section_key = f"{section} - מתמטי עם שם מקצוע"
            elif subject_type == 'הומני' and subject_name:
                section_key = f"{section} - הומני עם שם מקצוע"
            elif subject_type == 'מתמטי':
                section_key = f"{section} - מתמטי כללי"
            elif subject_type == 'הומני':
                section_key = f"{section} - הומני כללי"

        # Normalize to lowercase for cache lookup
        section_key = section_key.lower()

        # Map prompt types to file names - with aliases for backward compatibility
        prompt_files = {
            'free_chat': 'free_chat_prompt',
            'test_myself': 'test_myself_prompt'
        }

        if prompt_type not in prompt_files:
            logger.error(f"Unknown prompt type: {prompt_type}")
            return ""

        # Check if we need to reload or if not in cache
        if reload or prompt_type not in self._prompts_cache:
            filename = prompt_files[prompt_type]
            prompts = self._load_prompt_file(filename)
            if prompts:
                self._prompts_cache[prompt_type] = prompts
                logger.debug(f"Loaded {prompt_type} prompts from {filename}.md")
            else:
                logger.error(f"Failed to load {prompt_type} prompts from {filename}.md")
                return ""

        # Get prompts from cache
        prompts = self._prompts_cache.get(prompt_type, {})

        # Get the requested section using the constructed section_key
        prompt_text = prompts.get(section_key, '')

        # If not found with constructed key, try fallback to basic section name
        if not prompt_text and section_key != section.lower():
            prompt_text = prompts.get(section.lower(), '')
            if prompt_text:
                logger.debug(f"Found prompt using fallback section '{section.lower()}' instead of '{section_key}'")

        if not prompt_text:
            logger.warning(
                f"Section '{section_key}' not found in {prompt_type} prompts. Available sections: {list(prompts.keys())}")
            return ""

        # Format with provided variables if any
        if kwargs and prompt_text:
            try:
                prompt_text = prompt_text.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing variable {e} for prompt formatting")
            except Exception as e:
                logger.error(f"Error formatting prompt: {e}")

        return prompt_text

    def clear_cache(self):
        """Clear the prompts cache"""
        self._prompts_cache.clear()
        self._cache_loaded = False
        logger.info("Prompts cache cleared")

    def reload_prompts(self):
        """Reload all prompts from files"""
        self.clear_cache()
        self.preload_all_prompts()
        logger.info("All prompts reloaded from files")

    def get_cache_status(self) -> Dict[str, any]:
        """Get cache status information"""
        return {
            "cache_loaded": self._cache_loaded,
            "cached_prompt_types": list(self._prompts_cache.keys()),
            "cache_size": len(self._prompts_cache)
        }


# Global instance - use absolute path relative to project root
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
prompts_path = os.path.join(project_root, "Prompts")

# This will be replaced by app.state in FastAPI
_global_prompt_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get the global prompt loader instance"""
    global _global_prompt_loader
    if _global_prompt_loader is None:
        _global_prompt_loader = PromptLoader(prompts_path)
    return _global_prompt_loader


def initialize_prompt_loader() -> PromptLoader:
    """Initialize and preload prompt loader - for use in FastAPI startup"""
    global _global_prompt_loader
    _global_prompt_loader = PromptLoader(prompts_path)
    _global_prompt_loader.preload_all_prompts()
    return _global_prompt_loader


if __name__ == "__main__":
    # Test the prompt loader - focusing on the new prompts
    loader = PromptLoader(prompts_path)
    loader.preload_all_prompts()

    print("=" * 80)
    print("Testing Free Chat Prompts:")
    print("=" * 80)

    # Test basic free chat system prompt (without syllabus)
    print("1. Free Chat System Prompt (בלי סילבוס):")
    free_chat_system = loader.get_prompt("free_chat", "system")
    print(f"{free_chat_system if free_chat_system else 'NOT FOUND'}")

    print("\n" + "-" * 80)
    print("2. Free Chat System Prompt (עם סילבוס):")
    # Test free chat with syllabus
    free_chat_syllabus = loader.get_prompt("free_chat", "system - עם סילבוס",
                                           syllabus_content="זהו סילבוס לדוגמה של קורס מתמטיקה בדידה")
    print(f"{free_chat_syllabus if free_chat_syllabus else 'NOT FOUND'}")

    print("\n" + "=" * 80)
    print("Testing Test Myself Prompts:")
    print("=" * 80)

    # Test test_myself system prompt
    print("1. Test Myself System Prompt:")
    test_myself_system = loader.get_prompt("test_myself", "system")
    print(f"{test_myself_system if test_myself_system else 'NOT FOUND'}")

    print("\n" + "-" * 80)
    print("2. Test Myself User Prompt:")
    # Test test_myself user prompt
    test_myself_user = loader.get_prompt("test_myself", "user",
                                         conversation_context="זוהי תחילת השיחה.",
                                         context="מקור 1: יחס שקילות הוא יחס רפלקסיבי, סימטרי וטרנזיטיבי",
                                         query="מה זה יחס שקילות?")
    print(f"{test_myself_user if test_myself_user else 'NOT FOUND'}")

    print(f"\n" + "=" * 80)
    print(f"Cache Status: {loader.get_cache_status()}")
    print("Prompt Loader test completed successfully!")
