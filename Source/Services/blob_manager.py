"""
Blob Storage Manager
Manages upload and download of files to Azure Blob Storage
Supports all file types: MP4, MD, PDF, JSON etc.
"""

import os
from typing import Optional, List
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from Config.config import STORAGE_CONNECTION_STRING, CONTAINER_NAME
import traceback
import asyncio
from Config.logging_config import setup_logging
logger = setup_logging()

class BlobManager:
    """Manages upload and download of files to Azure Blob Storage"""

    def __init__(self, container_name: str = None):
        # If no container_name is passed, use default from config
        self.container_name = container_name if container_name is not None else CONTAINER_NAME

        # Initialize async client for reuse
        self._async_client = AsyncBlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

        # File type to content type mapping
        # File type to content type mapping
        self.content_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.md': 'text/markdown',
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.json': 'application/json',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.zip': 'application/zip',
            '.rar': 'application/x-rar-compressed',
            '.7z': 'application/x-7z-compressed'
        }

    def _get_content_type(self, file_path: str) -> str:
        """Determine content type based on file extension"""
        _, ext = os.path.splitext(file_path.lower())
        return self.content_types.get(ext, 'application/octet-stream')

    async def download_file(self, blob_name: str, local_file_path: str) -> bool:
        """
        Download file from blob storage (async)

        Args:
            blob_name: File name in blob storage (including folder if exists)
            local_file_path: Local save path

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            container_client = self._async_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)

            logger.info(f"Downloading file: {blob_name} -> {local_file_path}")

            stream = await blob_client.download_blob()
            file_bytes = await stream.readall()

            with open(local_file_path, 'wb') as file_data:
                file_data.write(file_bytes)

            logger.info(f"File downloaded successfully: {local_file_path}")
            return True

        except Exception as e:
            logger.info(f"Error downloading file {blob_name}: {e}")
            return False

    async def list_files(self, folder: Optional[str] = None) -> List[str]:
        """
        List files in blob storage (async)

        Args:
            folder: Specific folder (optional - if not specified will show all files)

        Returns:
            List of file names
        """
        try:
            container_client = self._async_client.get_container_client(self.container_name)

            name_filter = f"{folder}/" if folder else ""
            blob_list = []

            async for blob in container_client.list_blobs(name_starts_with=name_filter):
                blob_list.append(blob.name)

            return blob_list

        except Exception as e:
            logger.info(f"Error listing files: {e}")
            return []

    # def download_folder_files(self, blob_folder_path: str, local_temp_dir: str) -> List[str]:
    #     """
    #     הורדת כל הקבצים מתיקייה ב-blob storage לתיקייה מקומית זמנית
    #
    #     Args:
    #         blob_folder_path: נתיב התיקייה ב-blob storage (למשל: "Raw-data/Docs")
    #         local_temp_dir: תיקייה מקומית זמנית לשמירת הקבצים
    #
    #     Returns:
    #         רשימת נתיבי הקבצים המקומיים שהורדו
    #     """
    #     try:
    #         container_client = self.blob_service.get_container_client(self.container_name)
    #
    #         # יצירת התיקייה המקומית אם לא קיימת
    #         os.makedirs(local_temp_dir, exist_ok=True)
    #
    #         downloaded_files = []
    #
    #         logger.info(f"מוריד קבצים מ-blob: {blob_folder_path}")
    #
    #         # רשימת כל הקבצים בתיקייה
    #         blob_list = container_client.list_blobs(name_starts_with=blob_folder_path)
    #
    #         for blob in blob_list:
    #             # דילוג על תיקיות (שמות שמסתיימים ב-/)
    #             if blob.name.endswith('/'):
    #                 continue
    #
    #             # קבלת שם הקובץ בלבד (ללא הנתיב המלא)
    #             filename = os.path.basename(blob.name)
    #             local_file_path = os.path.join(local_temp_dir, filename)
    #
    #             logger.info(f"מוריד: {blob.name} → {local_file_path}")
    #
    #             # הורדת הקובץ
    #             blob_client = container_client.get_blob_client(blob.name)
    #             with open(local_file_path, "wb") as download_file:
    #                 download_file.write(blob_client.download_blob().readall())
    #
    #             downloaded_files.append(local_file_path)
    #
    #         logger.info(f"הורדו {len(downloaded_files)} קבצים מ-blob storage")
    #         return downloaded_files
    #
    #     except Exception as e:
    #         logger.info(f"שגיאה בהורדת קבצים מ-blob storage: {e}")
    #         return []

    async def generate_sas_url(self, blob_name: str, hours: int = 4) -> str:
        """
        Generate SAS URL for reading a file in blob storage (async)

        Args:
            blob_name: File name in blob storage (including folder if exists)
            hours: Number of hours the SAS will be valid (default: 4 hours)

        Returns:
            SAS URL for the file
        """
        try:
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self._async_client.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self._async_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=hours)
            )

            # Create full URL
            blob_url = f"{self._async_client.primary_endpoint}{self.container_name}/{blob_name}?{sas_token}"

            logger.info(f"Generated SAS URL for file: {blob_name} (valid for {hours} hours)")
            return blob_url

        except Exception as e:
            logger.info(f"Error generating SAS URL for {blob_name}: {e}")
            return ""

    async def download_to_memory(self, blob_name: str) -> Optional[bytes]:
        """
        Download file from blob storage directly to memory (async)

        Args:
            blob_name: File name in blob storage (including folder if exists)

        Returns:
            File bytes or None if failed
        """
        try:
            container_client = self._async_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)

            logger.info(f"Downloading file to memory: {blob_name}")

            stream = await blob_client.download_blob()
            file_bytes = await stream.readall()

            logger.info(f"File downloaded successfully to memory: {blob_name} ({len(file_bytes)} bytes)")
            return file_bytes

        except Exception as e:
            logger.info(f"Error downloading file to memory {blob_name}: {e}")
            return None

    async def upload_text_to_blob(self, text_content: str, blob_name: str, container: str = None) -> bool:
        """
        Upload text content directly to blob storage without creating temporary file (async)

        Args:
            text_content: Text content to upload
            blob_name: File name in blob (including virtual path)
            container: Container name (if not specified, will use default)

        Returns:
            True if upload succeeded, False otherwise
        """
        try:
            # Use passed container or default
            target_container = container if container else self.container_name

            # Determine content type based on file extension
            content_type = self._get_content_type(blob_name)

            container_client = self._async_client.get_container_client(target_container)

            logger.info(f"Uploading text to blob: {target_container}/{blob_name}")

            # Direct text upload
            await container_client.upload_blob(
                name=blob_name,
                data=text_content.encode('utf-8'),
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type)
            )

            logger.info(f"Text uploaded successfully: {target_container}/{blob_name}")
            return True

        except Exception as e:
            logger.info(f"Error uploading text to blob: {e}")
            return False

async def main():
    """Test the blob manager with async functions"""
    logger.info("Testing Blob Manager - Course Container")
    logger.info("=" * 50)

    try:
        blob_manager = BlobManager()

        # Check Section1 folder specifically
        logger.info("\nFiles in 'Section1' folder:")
        section1_blobs = await blob_manager.list_files("Section1")
        logger.info(f"Found {len(section1_blobs)} files in Section1:")
        for blob in section1_blobs:
            logger.info(f"  - {blob}")

        logger.info("\nAll files in container:")
        all_blobs = await blob_manager.list_files()
        logger.info(f"Total found {len(all_blobs)} files:")
        for blob in all_blobs[:10]:  # Show first 10
            logger.info(f"  - {blob}")
        if len(all_blobs) > 10:
            logger.info(f"  ... and {len(all_blobs) - 10} more files")

        # Test uploading text content
        logger.info("\nTesting text upload to Section1 folder:")
        test_content = "זהו קובץ בדיקה שהועלה ל-Section1\nתאריך: 2025-01-07"
        success = await blob_manager.upload_text_to_blob(test_content, "Section1/test_file.txt")

        if success:
            logger.info("Text uploaded successfully!")

            # List files again to see the new file
            logger.info("\nFiles in Section1 after upload:")
            updated_blobs = await blob_manager.list_files("Section1")
            for blob in updated_blobs:
                logger.info(f"  - {blob}")

        logger.info("\nBlob manager test completed!")

    except Exception as e:
        logger.info(f"Failed to test blob manager: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
