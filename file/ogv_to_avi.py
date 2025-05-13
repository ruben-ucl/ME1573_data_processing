#!/usr/bin/env python3
"""
OGV to Uncompressed AVI Converter

This script converts OGV video files to uncompressed AVI format for scientific data analysis.
It uses FFmpeg to perform the conversion with settings optimized for lossless output.

Usage:
    python ogv_to_uncompressed_avi.py input.ogv [output.avi]

If no output filename is provided, it will use the input filename with .avi extension.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def convert_ogv_to_uncompressed_avi(input_file, output_file=None):
    """
    Convert an OGV video file to uncompressed AVI format.

    Args:
        input_file (str): Path to the input OGV file
        output_file (str, optional): Path to the output AVI file.
            If not provided, uses the input filename with .avi extension.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    input_path = Path(input_file)
    
    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    if not input_path.suffix.lower() == '.ogv':
        print(f"Warning: Input file '{input_file}' does not have .ogv extension.")
    
    # Set output file if not provided
    if output_file is None:
        output_file = input_path.with_suffix('.avi')
    else:
        output_file = Path(output_file)
        if output_file.suffix.lower() != '.avi':
            output_file = output_file.with_suffix('.avi')
    
    # FFmpeg command for lossless conversion to uncompressed AVI
    # Using rawvideo codec for truly uncompressed video
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'rawvideo',    # Use rawvideo codec for uncompressed video
        '-pix_fmt', 'rgb24',   # RGB 24-bit color depth
        '-vsync', '0',         # Passthrough frame timestamps
        '-c:a', 'pcm_s16le',   # Uncompressed audio (if present)
        '-y',                  # Overwrite output file if it exists
        str(output_file)
    ]
    
    print(f"Converting '{input_file}' to uncompressed AVI format...")
    print(f"Output will be saved as '{output_file}'")
    print("Command:", " ".join(cmd))
    
    try:
        # Run the FFmpeg command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Conversion completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print(f"FFmpeg error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your PATH.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert OGV video to uncompressed AVI format.')
    parser.add_argument('input', help='Input OGV file path')
    parser.add_argument('output', nargs='?', help='Output AVI file path (optional)')
    args = parser.parse_args()
    
    convert_ogv_to_uncompressed_avi(args.input, args.output)

if __name__ == "__main__":
    main()