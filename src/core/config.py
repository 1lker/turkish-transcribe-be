"""
Configuration management for Turkish Education Transcription System
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from functools import lru_cache


class WhisperConfig(BaseModel):
    """Whisper model configuration"""
    model_config = ConfigDict(protected_namespaces=())  # Disable protected namespace warning
    
    model_size: str = Field(default="base", description="Model size to use")
    device: str = Field(default="cpu", description="Device to run on")
    compute_type: str = Field(default="float32", description="Compute type")
    language: str = Field(default="tr", description="Language code")
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt")
    temperature: float = Field(default=0.0, description="Temperature for sampling")
    beam_size: int = Field(default=5, description="Beam size for beam search")
    best_of: int = Field(default=5, description="Number of candidates")
    patience: float = Field(default=1.0, description="Patience for beam search")
    length_penalty: float = Field(default=1.0, description="Length penalty")
    suppress_tokens: List[int] = Field(default=[-1], description="Tokens to suppress")
    condition_on_previous_text: bool = Field(default=True, description="Condition on previous text")
    compression_ratio_threshold: float = Field(default=2.4, description="Compression ratio threshold")
    logprob_threshold: float = Field(default=-1.0, description="Log probability threshold")
    no_speech_threshold: float = Field(default=0.6, description="No speech threshold")
    
    @field_validator('model_size')
    @classmethod
    def validate_model_size(cls, v):
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        if v not in valid_sizes:
            raise ValueError(f"Model size must be one of {valid_sizes}")
        return v
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        if v not in ['cuda', 'cpu']:
            raise ValueError("Device must be 'cuda' or 'cpu'")
        return v


class AudioConfig(BaseModel):
    """Audio processing configuration"""
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_length: int = Field(default=30, description="Chunk length in seconds")
    overlap: int = Field(default=5, description="Overlap in seconds")
    format: str = Field(default="wav", description="Audio format")
    max_file_size: int = Field(default=5368709120, description="Max file size in bytes")
    allowed_formats: List[str] = Field(
        default=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav', 'm4a'],
        description="Allowed file formats"
    )


class ProcessingConfig(BaseModel):
    """Processing configuration"""
    batch_size: int = Field(default=4, description="Batch size for processing")
    max_concurrent_jobs: int = Field(default=2, description="Max concurrent jobs")
    timeout: int = Field(default=3600, description="Timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries")
    retry_delay: int = Field(default=5, description="Delay between retries")


class StorageConfig(BaseModel):
    """Storage configuration"""
    base_path: Path = Field(default=Path("./data"), description="Base storage path")
    raw_videos_path: Path = Field(default=Path("./data/raw"), description="Raw videos path")
    processed_path: Path = Field(default=Path("./data/processed"), description="Processed files path")
    transcripts_path: Path = Field(default=Path("./data/transcripts"), description="Transcripts path")
    temp_path: Path = Field(default=Path("./data/temp"), description="Temp files path")
    
    @property
    def raw_data_dir(self) -> Path:
        """Raw data directory path for backward compatibility"""
        return self.raw_videos_path
    
    def create_directories(self):
        """Create all necessary directories"""
        for path_field in ['base_path', 'raw_videos_path', 'processed_path', 'transcripts_path', 'temp_path']:
            path = getattr(self, path_field)
            path.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """Database configuration"""
    class PostgresConfig(BaseModel):
        host: str = "localhost"
        port: int = 5432
        database: str = "edu_transcription"
        username: str = "postgres"
        password: str = "postgres"
        echo: bool = False
        pool_size: int = 10
        max_overflow: int = 20
        
        @property
        def url(self) -> str:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class RedisConfig(BaseModel):
        host: str = "localhost"
        port: int = 6379
        db: int = 0
        password: Optional[str] = None
        decode_responses: bool = True
        
        @property
        def url(self) -> str:
            if self.password:
                return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
            return f"redis://{self.host}:{self.port}/{self.db}"
    
    class ChromaDBConfig(BaseModel):
        host: str = "localhost"
        port: int = 8001
        collection_name: str = "edu_transcripts"
        embedding_function: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format"
    )
    
    class FileConfig(BaseModel):
        enabled: bool = True
        path: Path = Path("./logs")
        rotation: str = "500 MB"
        retention: str = "10 days"
        compression: str = "zip"
    
    class ConsoleConfig(BaseModel):
        enabled: bool = True
        colorize: bool = True
        backtrace: bool = True
        diagnose: bool = True
    
    file: FileConfig = Field(default_factory=FileConfig)
    console: ConsoleConfig = Field(default_factory=ConsoleConfig)


class FeaturesConfig(BaseModel):
    """Feature flags configuration"""
    vad_enabled: bool = Field(default=True, description="Enable Voice Activity Detection")
    diarization_enabled: bool = Field(default=False, description="Enable speaker diarization")
    enhancement_enabled: bool = Field(default=True, description="Enable audio enhancement")
    auto_language_detection: bool = Field(default=False, description="Enable auto language detection")


class Config(BaseModel):
    """Main configuration class"""
    app: Dict[str, Any] = Field(default_factory=dict)
    server: Dict[str, Any] = Field(default_factory=dict)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse nested configurations
        if 'whisper' in config_dict:
            config_dict['whisper'] = WhisperConfig(**config_dict['whisper'])
        if 'audio' in config_dict:
            config_dict['audio'] = AudioConfig(**config_dict['audio'])
        if 'processing' in config_dict:
            config_dict['processing'] = ProcessingConfig(**config_dict['processing'])
        if 'storage' in config_dict:
            config_dict['storage'] = StorageConfig(**config_dict['storage'])
        if 'database' in config_dict:
            db_config = config_dict['database']
            config_dict['database'] = DatabaseConfig(
                postgres=DatabaseConfig.PostgresConfig(**db_config.get('postgres', {})),
                redis=DatabaseConfig.RedisConfig(**db_config.get('redis', {})),
                chromadb=DatabaseConfig.ChromaDBConfig(**db_config.get('chromadb', {}))
            )
        if 'logging' in config_dict:
            log_config = config_dict['logging']
            config_dict['logging'] = LoggingConfig(
                level=log_config.get('level', 'INFO'),
                format=log_config.get('format', LoggingConfig().format),
                file=LoggingConfig.FileConfig(**log_config.get('file', {})),
                console=LoggingConfig.ConsoleConfig(**log_config.get('console', {}))
            )
        if 'features' in config_dict:
            config_dict['features'] = FeaturesConfig(**config_dict['features'])
        
        return cls(**config_dict)
    
    def save(self, config_path: Path):
        """Save configuration to YAML file"""
        config_dict = self.model_dump()
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def setup_environment(self):
        """Setup environment based on configuration"""
        # Create necessary directories
        self.storage.create_directories()
        
        # Set environment variables
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' if self.whisper.device == 'cuda' else ''
        
        return self


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env
    )
    
    config_path: Path = Field(
        default=Path("configs/config.yaml"),
        description="Path to configuration file"
    )
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def get_config() -> Config:
    """Get cached configuration instance"""
    settings = get_settings()
    config = Config.from_yaml(settings.config_path)
    config.setup_environment()
    return config


# Global instances
settings = get_settings()
config = get_config()


if __name__ == "__main__":
    # Test configuration loading
    print("Loading configuration...")
    test_config = get_config()
    print(f"App Name: {test_config.app.get('name')}")
    print(f"Whisper Model: {test_config.whisper.model_size}")
    print(f"Storage Path: {test_config.storage.base_path}")
    print("Configuration loaded successfully!")