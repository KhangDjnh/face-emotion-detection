"""
Main script để chạy inference pipeline.
Usage:
    python run_inference.py                    # Webcam
    python run_inference.py video.mp4          # Video file
    python run_inference.py video.mp4 output.json  # Video file với output JSON
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.inference.pipeline import run_realtime_inference

if __name__ == "__main__":
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    run_realtime_inference(video_source, output_file)

