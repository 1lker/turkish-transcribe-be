"""
Logging configuration for Turkish Education Transcription System
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger as loguru_logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


# Rich console for pretty output
console = Console()


class TranscriptionLogger:
    """Custom logger wrapper with rich output support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize logger with configuration"""
        self.config = config or {}
        self.setup_logger()
        self.progress_bars = {}
        
    def setup_logger(self):
        """Setup loguru logger with custom configuration"""
        # Remove default logger
        loguru_logger.remove()
        
        # Get logging config
        log_config = self.config.get('logging', {})
        level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', 
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        # Console logging
        console_config = log_config.get('console', {})
        if console_config.get('enabled', True):
            loguru_logger.add(
                sys.stderr,
                format=log_format,
                level=level,
                colorize=console_config.get('colorize', True),
                backtrace=console_config.get('backtrace', True),
                diagnose=console_config.get('diagnose', True),
                enqueue=True
            )
        
        # File logging
        file_config = log_config.get('file', {})
        if file_config.get('enabled', True):
            log_path = Path(file_config.get('path', './logs'))
            log_path.mkdir(parents=True, exist_ok=True)
            
            loguru_logger.add(
                log_path / "transcription_{time:YYYY-MM-DD}.log",
                format=log_format,
                level=level,
                rotation=file_config.get('rotation', '500 MB'),
                retention=file_config.get('retention', '10 days'),
                compression=file_config.get('compression', 'zip'),
                enqueue=True,
                encoding='utf-8'
            )
            
            # Separate error log
            loguru_logger.add(
                log_path / "errors_{time:YYYY-MM-DD}.log",
                format=log_format,
                level="ERROR",
                rotation=file_config.get('rotation', '500 MB'),
                retention=file_config.get('retention', '30 days'),
                compression=file_config.get('compression', 'zip'),
                enqueue=True,
                encoding='utf-8',
                backtrace=True,
                diagnose=True
            )
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        loguru_logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        loguru_logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        loguru_logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        loguru_logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        loguru_logger.critical(message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message with rich formatting"""
        loguru_logger.success(message, **kwargs)
        console.print(f"[bold green]✓[/bold green] {message}")
    
    def print_banner(self):
        """Print application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║     Turkish Education Transcription System v1.0.0            ║
║     🎓 Powered by Whisper AI | 🇹🇷 Optimized for Turkish      ║
╚══════════════════════════════════════════════════════════════╝
        """
        console.print(Panel.fit(banner, style="bold cyan"))
    
    def print_config(self, config: Dict[str, Any]):
        """Print configuration in a nice table"""
        table = Table(title="System Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        # Flatten config for display
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    table.add_row(f"{section}.{key}", str(value))
            else:
                table.add_row(section, str(values))
        
        console.print(table)
    
    def create_progress(self, task_name: str, total: int) -> Progress:
        """Create a progress bar for long running tasks"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            transient=True
        )
        task_id = progress.add_task(task_name, total=total)
        self.progress_bars[task_name] = (progress, task_id)
        progress.start()
        return progress
    
    def update_progress(self, task_name: str, advance: int = 1):
        """Update progress bar"""
        if task_name in self.progress_bars:
            progress, task_id = self.progress_bars[task_name]
            progress.advance(task_id, advance)
    
    def complete_progress(self, task_name: str):
        """Complete and remove progress bar"""
        if task_name in self.progress_bars:
            progress, task_id = self.progress_bars[task_name]
            progress.stop()
            del self.progress_bars[task_name]
    
    def log_transcription_start(self, file_path: str, metadata: Dict[str, Any]):
        """Log transcription start with details"""
        console.print("\n" + "="*60)
        console.print(f"[bold cyan]🎬 Starting Transcription[/bold cyan]")
        console.print(f"[yellow]File:[/yellow] {file_path}")
        
        if metadata:
            table = Table(show_header=False, box=None)
            table.add_column("Property", style="dim")
            table.add_column("Value", style="green")
            
            for key, value in metadata.items():
                table.add_row(key, str(value))
            
            console.print(table)
        console.print("="*60 + "\n")
    
    def log_transcription_complete(self, file_path: str, duration: float, stats: Dict[str, Any]):
        """Log transcription completion with statistics"""
        console.print("\n" + "="*60)
        console.print(f"[bold green]✅ Transcription Complete[/bold green]")
        console.print(f"[yellow]File:[/yellow] {file_path}")
        console.print(f"[yellow]Duration:[/yellow] {duration:.2f} seconds")
        
        if stats:
            table = Table(title="Statistics", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in stats.items():
                table.add_row(key, str(value))
            
            console.print(table)
        console.print("="*60 + "\n")
    
    def log_error_details(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log detailed error information"""
        console.print(f"\n[bold red]❌ Error Occurred[/bold red]")
        console.print(f"[red]Type:[/red] {type(error).__name__}")
        console.print(f"[red]Message:[/red] {str(error)}")
        
        if context:
            console.print("\n[yellow]Context:[/yellow]")
            for key, value in context.items():
                console.print(f"  {key}: {value}")
        
        loguru_logger.exception(error)
    
    def log_json(self, data: Dict[str, Any], title: str = "JSON Data"):
        """Log JSON data in a formatted way"""
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        console.print(Panel(json_str, title=title, style="cyan"))
        loguru_logger.debug(f"{title}: {json_str}")


def setup_logger(config: Optional[Dict[str, Any]] = None) -> TranscriptionLogger:
    """Setup and return a configured logger instance"""
    return TranscriptionLogger(config)


# Create default logger instance
default_logger = TranscriptionLogger()


# Module-level functions that use the default logger
def info(message: str, **kwargs):
    default_logger.info(message, **kwargs)

def debug(message: str, **kwargs):
    default_logger.debug(message, **kwargs)

def warning(message: str, **kwargs):
    default_logger.warning(message, **kwargs)

def error(message: str, **kwargs):
    default_logger.error(message, **kwargs)

def critical(message: str, **kwargs):
    default_logger.critical(message, **kwargs)

def success(message: str, **kwargs):
    default_logger.success(message, **kwargs)


# Export the default logger
logger = default_logger


def get_logger(name: str = __name__) -> TranscriptionLogger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module/logger
        
    Returns:
        TranscriptionLogger instance
    """
    return default_logger


if __name__ == "__main__":
    # Test the logger
    test_logger = setup_logger()
    
    test_logger.print_banner()
    
    test_logger.info("This is an info message")
    test_logger.debug("This is a debug message")
    test_logger.warning("This is a warning message")
    test_logger.success("This is a success message")
    
    # Test progress bar
    import time
    progress = test_logger.create_progress("Processing files", 10)
    for i in range(10):
        time.sleep(0.5)
        test_logger.update_progress("Processing files")
    test_logger.complete_progress("Processing files")
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        test_logger.log_error_details(e, {"file": "test.mp4", "stage": "testing"})
    
    # Test JSON logging
    test_data = {
        "model": "whisper-base",
        "language": "tr",
        "accuracy": 0.95
    }
    test_logger.log_json(test_data, "Model Configuration")
    
    print("\nLogger test completed!")