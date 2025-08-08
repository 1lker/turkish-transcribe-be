#!/usr/bin/env python3
"""
CLI Interface for Turkish Education Transcription System
"""

import sys
import click
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import logger
from src.core.config import config
from src.transcription import TranscriptionPipeline, WhisperEngine, AudioProcessor

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="Turkish Edu Transcription")
def cli():
    """
    🎓 Turkish Education Transcription System
    
    Powered by Whisper AI | Optimized for Turkish
    """
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--model', '-m', default='base', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--device', '-d', default='auto',
              type=click.Choice(['auto', 'cpu', 'cuda']),
              help='Device to use')
@click.option('--language', '-l', default='tr', help='Language code')
@click.option('--vad/--no-vad', default=True, help='Apply voice activity detection')
@click.option('--normalize/--no-normalize', default=True, help='Normalize audio')
@click.option('--format', '-f', 
              type=click.Choice(['json', 'txt', 'srt', 'all']),
              default='all', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def transcribe(input_file, output, model, device, language, vad, normalize, format, verbose):
    """
    Transcribe a single audio or video file
    
    Example:
        python cli.py transcribe video.mp4 -o output/ -m base -l tr
    """
    logger.print_banner()
    
    input_path = Path(input_file)
    output_dir = Path(output) if output else config.storage.transcripts_path / input_path.stem
    
    # Show configuration
    if verbose:
        config_table = Table(title="Transcription Configuration", show_header=True)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        config_table.add_row("Input File", str(input_path))
        config_table.add_row("Output Directory", str(output_dir))
        config_table.add_row("Model", model)
        config_table.add_row("Device", device)
        config_table.add_row("Language", language)
        config_table.add_row("VAD", "✓" if vad else "✗")
        config_table.add_row("Normalize", "✓" if normalize else "✗")
        config_table.add_row("Output Format", format)
        console.print(config_table)
    
    try:
        # Initialize pipeline
        with console.status("[bold green]Initializing pipeline...") as status:
            pipeline = TranscriptionPipeline(
                model_size=model,
                device=device if device != 'auto' else None
            )
        
        # Process file
        console.print("\n[bold cyan]Processing file...[/bold cyan]")
        result = pipeline.process_file(
            input_path,
            output_dir=output_dir,
            language=language,
            apply_vad=vad,
            normalize_audio=normalize,
            save_intermediate=verbose
        )
        
        if result['success']:
            # Display results
            console.print("\n[bold green]✅ Transcription Complete![/bold green]")
            
            # Show transcript preview
            transcript = result['result'].text
            preview_length = 500
            if len(transcript) > preview_length:
                preview = transcript[:preview_length] + "..."
            else:
                preview = transcript
            
            console.print(Panel(preview, title="Transcript Preview", style="green"))
            
            # Show statistics
            metadata = result['metadata']
            stats_table = Table(title="Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            stats_table.add_row("Duration", f"{metadata['original_duration']:.2f} seconds")
            stats_table.add_row("Processing Time", f"{metadata['processing_time']:.2f} seconds")
            stats_table.add_row("Real-time Factor", f"{metadata['real_time_factor']:.2f}x")
            stats_table.add_row("Word Count", str(metadata['word_count']))
            stats_table.add_row("Character Count", str(metadata['character_count']))
            stats_table.add_row("Segments", str(metadata['segment_count']))
            console.print(stats_table)
            
            # Show output files
            console.print("\n[bold]Output Files:[/bold]")
            for file_type, file_path in metadata['output_files'].items():
                if format == 'all' or format == file_type:
                    console.print(f"  • {file_type.upper()}: [green]{file_path}[/green]")
        
        else:
            console.print(f"\n[bold red]❌ Transcription Failed![/bold red]")
            console.print(f"Error: {result['error']}")
            sys.exit(1)
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--model', '-m', default='base',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--pattern', '-p', default='*', help='File pattern to match')
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories')
@click.option('--parallel/--sequential', default=True, help='Process files in parallel')
@click.option('--batch-size', '-b', default=4, help='Batch size for parallel processing')
@click.option('--language', '-l', default='tr', help='Language code')
def batch(input_dir, output, model, pattern, recursive, parallel, batch_size, language):
    """
    Batch transcribe multiple files in a directory
    
    Example:
        python cli.py batch videos/ -o output/ -r -p "*.mp4"
    """
    logger.print_banner()
    
    input_path = Path(input_dir)
    output_dir = Path(output) if output else config.storage.transcripts_path / input_path.name
    
    console.print(f"[bold cyan]Batch Processing Directory:[/bold cyan] {input_path}")
    console.print(f"[bold cyan]Output Directory:[/bold cyan] {output_dir}")
    console.print(f"[bold cyan]Pattern:[/bold cyan] {pattern}")
    console.print(f"[bold cyan]Recursive:[/bold cyan] {'Yes' if recursive else 'No'}")
    console.print(f"[bold cyan]Parallel:[/bold cyan] {'Yes' if parallel else 'No'}")
    
    try:
        # Initialize pipeline
        with console.status("[bold green]Initializing pipeline...") as status:
            pipeline = TranscriptionPipeline(
                model_size=model,
                batch_size=batch_size
            )
        
        # Process directory
        console.print("\n[bold cyan]Processing files...[/bold cyan]")
        summary = pipeline.process_directory(
            input_path,
            output_dir=output_dir,
            recursive=recursive,
            file_pattern=pattern,
            parallel=parallel,
            language=language
        )
        
        # Display summary
        console.print("\n[bold green]✅ Batch Processing Complete![/bold green]")
        
        summary_table = Table(title="Processing Summary", show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        summary_table.add_row("Total Files", str(summary['total_files']))
        summary_table.add_row("Processed", str(summary['processed']))
        summary_table.add_row("Errors", str(summary['errors']))
        summary_table.add_row("Output Directory", str(summary['output_dir']))
        console.print(summary_table)
        
        # Show statistics
        stats = summary['statistics']
        if stats['total_files_processed'] > 0:
            stats_table = Table(title="Performance Statistics", show_header=False)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="yellow")
            stats_table.add_row("Total Duration", f"{stats['total_duration_hours']:.2f} hours")
            stats_table.add_row("Processing Time", f"{stats['total_processing_time_hours']:.2f} hours")
            stats_table.add_row("Avg Real-time Factor", f"{stats['average_real_time_factor']:.2f}x")
            console.print(stats_table)
        
        # Cleanup
        pipeline.cleanup()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
def detect_language(audio_file):
    """
    Detect the language of an audio file
    
    Example:
        python cli.py detect-language audio.wav
    """
    logger.print_banner()
    
    audio_path = Path(audio_file)
    console.print(f"[bold cyan]Detecting language for:[/bold cyan] {audio_path}")
    
    try:
        # Initialize engine
        with console.status("[bold green]Loading model...") as status:
            engine = WhisperEngine(model_size="base")
        
        # Detect language
        with console.status("[bold green]Analyzing audio...") as status:
            language, confidence = engine.detect_language(audio_path)
        
        # Display results
        console.print("\n[bold green]Language Detection Results:[/bold green]")
        console.print(f"Language: [bold yellow]{language}[/bold yellow]")
        console.print(f"Confidence: [bold yellow]{confidence:.2%}[/bold yellow]")
        
        # Language name mapping
        language_names = {
            'tr': 'Turkish',
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'ar': 'Arabic',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean'
        }
        
        if language in language_names:
            console.print(f"Language Name: [bold yellow]{language_names[language]}[/bold yellow]")
        
        # Cleanup
        engine.unload_model()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output audio file')
@click.option('--vad/--no-vad', default=True, help='Apply voice activity detection')
@click.option('--normalize/--no-normalize', default=True, help='Normalize audio')
@click.option('--format', '-f', default='wav', 
              type=click.Choice(['wav', 'mp3', 'flac']),
              help='Output audio format')
def preprocess(input_file, output, vad, normalize, format):
    """
    Preprocess audio/video file for transcription
    
    Example:
        python cli.py preprocess video.mp4 -o clean_audio.wav --vad
    """
    logger.print_banner()
    
    input_path = Path(input_file)
    console.print(f"[bold cyan]Preprocessing:[/bold cyan] {input_path}")
    
    try:
        # Initialize processor
        processor = AudioProcessor()
        
        # Get original info
        console.print("\n[bold]Original File Info:[/bold]")
        info = processor.get_audio_info(input_path)
        info_table = Table(show_header=False)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="yellow")
        info_table.add_row("Duration", f"{info.duration_minutes:.2f} minutes")
        info_table.add_row("Sample Rate", f"{info.sample_rate} Hz")
        info_table.add_row("Channels", str(info.channels))
        info_table.add_row("Size", f"{info.file_size_mb:.2f} MB")
        console.print(info_table)
        
        # Process audio
        with console.status("[bold green]Processing audio...") as status:
            processed = processor.process_for_transcription(
                input_path,
                apply_vad=vad,
                normalize=normalize
            )
        
        # Convert to desired format if needed
        if output:
            output_path = Path(output)
            if output_path.suffix[1:] != format:
                output_path = output_path.with_suffix(f'.{format}')
            
            if processed != output_path:
                with console.status(f"[bold green]Converting to {format}...") as status:
                    final_path = processor.convert_audio_format(
                        processed,
                        output_format=format,
                        output_path=output_path
                    )
            else:
                final_path = processed
        else:
            final_path = processed
        
        # Show results
        console.print("\n[bold green]✅ Preprocessing Complete![/bold green]")
        console.print(f"Output: [green]{final_path}[/green]")
        
        # Get processed info
        processed_info = processor.get_audio_info(final_path)
        if vad and processed_info.duration < info.duration:
            reduction = (1 - processed_info.duration / info.duration) * 100
            console.print(f"Duration Reduction: [yellow]{reduction:.1f}%[/yellow]")
        
        # Cleanup
        processor.cleanup_temp_files()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True))
def info(audio_file):
    """
    Get information about an audio/video file
    
    Example:
        python cli.py info audio.wav
    """
    audio_path = Path(audio_file)
    console.print(f"[bold cyan]File Information:[/bold cyan] {audio_path}")
    
    try:
        # Initialize processor
        processor = AudioProcessor()
        
        # Get info
        file_info = processor.get_audio_info(audio_path)
        
        # Display info
        info_table = Table(title="Audio File Information", show_header=False)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="yellow")
        info_table.add_row("File", str(file_info.file_path.name))
        info_table.add_row("Format", file_info.format)
        info_table.add_row("Codec", file_info.codec or "Unknown")
        info_table.add_row("Duration", f"{file_info.duration:.2f} seconds")
        info_table.add_row("Duration (minutes)", f"{file_info.duration_minutes:.2f} minutes")
        info_table.add_row("Sample Rate", f"{file_info.sample_rate} Hz")
        info_table.add_row("Channels", str(file_info.channels))
        info_table.add_row("Bit Rate", f"{file_info.bit_rate:,} bps" if file_info.bit_rate else "Unknown")
        info_table.add_row("File Size", f"{file_info.file_size_mb:.2f} MB")
        console.print(info_table)
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
def models():
    """
    List available Whisper models and their sizes
    """
    console.print("[bold cyan]Available Whisper Models:[/bold cyan]\n")
    
    models_data = [
        ("tiny", "39M", "~1GB", "~10x", "Fast, lower quality"),
        ("base", "74M", "~1GB", "~7x", "Good balance"),
        ("small", "244M", "~2GB", "~4x", "Better quality"),
        ("medium", "769M", "~5GB", "~2x", "High quality"),
        ("large", "1550M", "~10GB", "~1x", "Best quality"),
    ]
    
    table = Table(title="Whisper Models", show_header=True)
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", style="yellow")
    table.add_column("VRAM", style="green")
    table.add_column("Speed", style="magenta")
    table.add_column("Description", style="white")
    
    for model_info in models_data:
        table.add_row(*model_info)
    
    console.print(table)
    console.print("\n[dim]Note: Speed is relative to real-time on CPU. GPU is typically 2-5x faster.[/dim]")


@cli.command()
def config_info():
    """
    Show current configuration
    """
    console.print("[bold cyan]Current Configuration:[/bold cyan]\n")
    
    # Load config
    from src.core.config import config
    
    # Display configuration sections
    sections = {
        "App": config.app,
        "Whisper": {
            "model_size": config.whisper.model_size,
            "device": config.whisper.device,
            "language": config.whisper.language,
            "beam_size": config.whisper.beam_size,
        },
        "Audio": {
            "sample_rate": config.audio.sample_rate,
            "chunk_length": config.audio.chunk_length,
            "max_file_size_mb": config.audio.max_file_size / (1024**2),
            "allowed_formats": ", ".join(config.audio.allowed_formats),
        },
        "Processing": {
            "batch_size": config.processing.batch_size,
            "max_concurrent_jobs": config.processing.max_concurrent_jobs,
            "timeout": config.processing.timeout,
        },
        "Features": {
            "vad_enabled": config.features.vad_enabled,
            "diarization_enabled": config.features.diarization_enabled,
            "enhancement_enabled": config.features.enhancement_enabled,
        }
    }
    
    for section_name, section_data in sections.items():
        table = Table(title=section_name, show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        
        for key, value in section_data.items():
            table.add_row(key, str(value))
        
        console.print(table)
        console.print()


if __name__ == "__main__":
    cli()