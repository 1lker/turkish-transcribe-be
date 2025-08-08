"""
Custom exceptions for Turkish Education Transcription System
"""

from typing import Optional, Dict, Any


class TranscriptionError(Exception):
    """Base exception for transcription related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class AudioProcessingError(TranscriptionError):
    """Exception raised for audio processing errors"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 stage: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.file_path = file_path
        self.stage = stage
        
        if details is None:
            details = {}
        
        if file_path:
            details['file_path'] = file_path
        if stage:
            details['stage'] = stage
        
        super().__init__(message, details)


class ModelLoadError(TranscriptionError):
    """Exception raised when model loading fails"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 model_size: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_size = model_size
        
        if details is None:
            details = {}
        
        if model_name:
            details['model_name'] = model_name
        if model_size:
            details['model_size'] = model_size
        
        super().__init__(message, details)


class ValidationError(TranscriptionError):
    """Exception raised for validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Any = None, details: Optional[Dict[str, Any]] = None):
        self.field = field
        self.value = value
        
        if details is None:
            details = {}
        
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = value
        
        super().__init__(message, details)


class StorageError(TranscriptionError):
    """Exception raised for storage related errors"""
    
    def __init__(self, message: str, path: Optional[str] = None, 
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.path = path
        self.operation = operation
        
        if details is None:
            details = {}
        
        if path:
            details['path'] = path
        if operation:
            details['operation'] = operation
        
        super().__init__(message, details)


class FileFormatError(ValidationError):
    """Exception raised for unsupported file formats"""
    
    def __init__(self, message: str, file_format: Optional[str] = None, 
                 supported_formats: Optional[list] = None, details: Optional[Dict[str, Any]] = None):
        self.file_format = file_format
        self.supported_formats = supported_formats
        
        if details is None:
            details = {}
        
        if file_format:
            details['file_format'] = file_format
        if supported_formats:
            details['supported_formats'] = supported_formats
        
        super().__init__(message, details=details)


class FileSizeError(ValidationError):
    """Exception raised when file size exceeds limit"""
    
    def __init__(self, message: str, file_size: Optional[int] = None, 
                 max_size: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.file_size = file_size
        self.max_size = max_size
        
        if details is None:
            details = {}
        
        if file_size:
            details['file_size'] = file_size
            details['file_size_mb'] = file_size / (1024 * 1024)
        if max_size:
            details['max_size'] = max_size
            details['max_size_mb'] = max_size / (1024 * 1024)
        
        super().__init__(message, details=details)


class TranscriptionTimeoutError(TranscriptionError):
    """Exception raised when transcription times out"""
    
    def __init__(self, message: str, timeout_seconds: Optional[int] = None, 
                 elapsed_seconds: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        
        if details is None:
            details = {}
        
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds
        if elapsed_seconds:
            details['elapsed_seconds'] = elapsed_seconds
        
        super().__init__(message, details)


class GPUError(TranscriptionError):
    """Exception raised for GPU related errors"""
    
    def __init__(self, message: str, gpu_available: bool = False, 
                 cuda_version: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.gpu_available = gpu_available
        self.cuda_version = cuda_version
        
        if details is None:
            details = {}
        
        details['gpu_available'] = gpu_available
        if cuda_version:
            details['cuda_version'] = cuda_version
        
        super().__init__(message, details)


class DatabaseError(TranscriptionError):
    """Exception raised for database operations"""
    
    def __init__(self, message: str, database_type: Optional[str] = None,
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.database_type = database_type
        self.operation = operation
        
        if details is None:
            details = {}
        
        if database_type:
            details['database_type'] = database_type
        if operation:
            details['operation'] = operation
        
        super().__init__(message, details)


class QueueError(TranscriptionError):
    """Exception raised for queue operations"""
    
    def __init__(self, message: str, queue_size: Optional[int] = None,
                 max_size: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.queue_size = queue_size
        self.max_size = max_size
        
        if details is None:
            details = {}
        
        if queue_size is not None:
            details['queue_size'] = queue_size
        if max_size is not None:
            details['max_size'] = max_size
        
        super().__init__(message, details)


class ConfigurationError(TranscriptionError):
    """Exception raised for configuration errors"""
    
    def __init__(self, message: str, config_file: Optional[str] = None,
                 missing_field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.config_file = config_file
        self.missing_field = missing_field
        
        if details is None:
            details = {}
        
        if config_file:
            details['config_file'] = config_file
        if missing_field:
            details['missing_field'] = missing_field
        
        super().__init__(message, details)


def handle_exception(func):
    """Decorator to handle exceptions in functions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TranscriptionError:
            # Re-raise our custom exceptions
            raise
        except FileNotFoundError as e:
            raise StorageError(f"File not found: {str(e)}", operation="read")
        except PermissionError as e:
            raise StorageError(f"Permission denied: {str(e)}", operation="access")
        except MemoryError as e:
            raise TranscriptionError(f"Out of memory: {str(e)}")
        except Exception as e:
            # Wrap unexpected exceptions
            raise TranscriptionError(f"Unexpected error: {str(e)}")
    
    return wrapper


class YouTubeDownloadError(TranscriptionError):
    """Exception raised for YouTube download errors"""
    
    def __init__(self, message: str, url: Optional[str] = None, 
                 video_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.url = url
        self.video_id = video_id
        
        if details is None:
            details = {}
        
        if url:
            details['url'] = url
        if video_id:
            details['video_id'] = video_id
        
        super().__init__(message, details)


class InvalidURLError(TranscriptionError):
    """Exception raised for invalid URL errors"""
    
    def __init__(self, message: str, url: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        self.url = url
        
        if details is None:
            details = {}
        
        if url:
            details['url'] = url
        
        super().__init__(message, details)


if __name__ == "__main__":
    # Test exceptions
    print("Testing custom exceptions...")
    
    try:
        raise AudioProcessingError(
            "Failed to process audio", 
            file_path="/path/to/file.mp4",
            stage="preprocessing"
        )
    except AudioProcessingError as e:
        print(f"AudioProcessingError: {e}")
        print(f"Details: {e.details}")
    
    try:
        raise FileSizeError(
            "File too large",
            file_size=10737418240,  # 10GB
            max_size=5368709120     # 5GB
        )
    except FileSizeError as e:
        print(f"\nFileSizeError: {e}")
        print(f"Details: {e.details}")
    
    try:
        raise ModelLoadError(
            "Failed to load Whisper model",
            model_name="whisper",
            model_size="large-v3"
        )
    except ModelLoadError as e:
        print(f"\nModelLoadError: {e}")
        print(f"Details: {e.details}")
    
    print("\nException tests completed!")