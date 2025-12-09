"""
Field Boundary Detection using Gemini 3 Pro Image Preview

This script processes satellite images to detect and highlight agricultural 
field boundaries using Google's Gemini 3 Pro Image Preview model.

Secure API Key Options:
1. Set GOOGLE_API_KEY environment variable
2. Create a .env file with GOOGLE_API_KEY=your_key
3. Enter interactively when prompted (secure - not echoed)
"""

import os
import sys
import argparse
import getpass
import time
from pathlib import Path
from typing import Optional, List

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    from tqdm import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

import config


def get_api_key() -> str:
    """
    Get API key securely with multiple fallback options.
    
    Priority:
    1. Environment variable GOOGLE_API_KEY
    2. .env file
    3. Interactive secure prompt (key not echoed)
    
    Returns:
        str: The API key
    """
    # Load .env file if it exists
    load_dotenv()
    
    # Check environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key and api_key != "your_api_key_here":
        print("✓ API key loaded from environment")
        return api_key
    
    # Interactive secure prompt
    print("\n" + "="*60)
    print("API Key Required")
    print("="*60)
    print("Your API key will NOT be displayed as you type.")
    print("Get your key from: https://aistudio.google.com/apikey")
    print("-"*60)
    
    api_key = getpass.getpass("Enter your Google AI API key: ")
    
    if not api_key:
        print("Error: No API key provided.")
        sys.exit(1)
    
    # Offer to save for future use
    save_choice = input("\nSave API key to .env file for future use? (y/n): ").lower().strip()
    if save_choice == 'y':
        env_path = Path(__file__).parent / ".env"
        with open(env_path, 'w') as f:
            f.write(f"GOOGLE_API_KEY={api_key}\n")
        print(f"✓ API key saved to {env_path}")
        print("  (This file is in .gitignore and won't be committed)")
    
    return api_key


class FieldBoundaryDetector:
    """Detect and highlight agricultural field boundaries in satellite images."""
    
    def __init__(self, api_key: str, resolution: str = None, prompt: str = None):
        """
        Initialize the detector.
        
        Args:
            api_key: Google AI API key
            resolution: Output resolution ("1K", "2K", or "4K")
            prompt: Custom prompt for boundary detection
        """
        self.client = genai.Client(api_key=api_key)
        self.resolution = resolution or config.DEFAULT_RESOLUTION
        self.prompt = prompt or config.DEFAULT_PROMPT
        self.model = config.MODEL_NAME
        
        print(f"✓ Initialized with model: {self.model}")
        print(f"✓ Output resolution: {self.resolution}")
    
    def process_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Process a single satellite image to detect field boundaries.
        
        Args:
            image_path: Path to input satellite image
            output_path: Path to save processed image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Determine aspect ratio from original image
            width, height = image.size
            aspect_ratio = self._get_closest_aspect_ratio(width, height)
            
            # Make API request with retry logic
            for attempt in range(config.MAX_RETRIES):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[self.prompt, image],
                        config=types.GenerateContentConfig(
                            response_modalities=['IMAGE'],
                            image_config=types.ImageConfig(
                                aspect_ratio=aspect_ratio,
                                image_size=self.resolution
                            ),
                        )
                    )
                    break
                except Exception as e:
                    if attempt < config.MAX_RETRIES - 1:
                        print(f"  Retry {attempt + 1}/{config.MAX_RETRIES} after error: {e}")
                        time.sleep(config.RETRY_DELAY_SECONDS)
                    else:
                        raise
            
            # Save the result
            for part in response.parts:
                if part.inline_data is not None:
                    result_image = part.as_image()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    result_image.save(output_path)
                    return True
                elif part.text is not None:
                    print(f"  Model response: {part.text}")
            
            print(f"  Warning: No image in response for {image_path.name}")
            return False
            
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            return False
    
    def _get_closest_aspect_ratio(self, width: int, height: int) -> str:
        """Get the closest supported aspect ratio for the image dimensions."""
        ratio = width / height
        
        # Supported ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
        ratios = {
            "1:1": 1.0,
            "2:3": 2/3,
            "3:2": 3/2,
            "3:4": 3/4,
            "4:3": 4/3,
            "4:5": 4/5,
            "5:4": 5/4,
            "9:16": 9/16,
            "16:9": 16/9,
            "21:9": 21/9,
        }
        
        closest = min(ratios.items(), key=lambda x: abs(x[1] - ratio))
        return closest[0]
    
    def process_batch(self, input_dir: Path, output_dir: Path) -> tuple:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            
        Returns:
            tuple: (successful_count, failed_count)
        """
        # Find all supported images
        images: List[Path] = []
        for fmt in config.SUPPORTED_FORMATS:
            images.extend(input_dir.glob(f"*{fmt}"))
            images.extend(input_dir.glob(f"*{fmt.upper()}"))
        
        if not images:
            print(f"No images found in {input_dir}")
            return 0, 0
        
        print(f"\nProcessing {len(images)} images...")
        print("-" * 60)
        
        successful = 0
        failed = 0
        
        for image_path in tqdm(images, desc="Processing"):
            # Create output filename
            output_name = f"{image_path.stem}{config.OUTPUT_SUFFIX}{image_path.suffix}"
            output_path = output_dir / output_name
            
            if self.process_image(image_path, output_path):
                successful += 1
            else:
                failed += 1
        
        print("-" * 60)
        print(f"Complete: {successful} successful, {failed} failed")
        
        return successful, failed


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Detect and highlight agricultural field boundaries in satellite images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python field_boundary_detector.py --input satellite.jpg --output result.png
  
  # Batch process directory with 4K output
  python field_boundary_detector.py --input-dir ./images --output-dir ./processed --resolution 4K
  
  # Use custom prompt
  python field_boundary_detector.py --input image.jpg --prompt "Highlight all farm boundaries"
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=Path, help="Single input image path")
    input_group.add_argument("--input-dir", type=Path, help="Directory of images for batch processing")
    
    # Output options
    parser.add_argument("--output", "-o", type=Path, help="Output path for single image")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), 
                        help="Output directory for batch processing (default: ./output)")
    
    # Processing options
    parser.add_argument("--resolution", "-r", choices=["1K", "2K", "4K"], 
                        default=config.DEFAULT_RESOLUTION,
                        help=f"Output resolution (default: {config.DEFAULT_RESOLUTION})")
    parser.add_argument("--prompt", "-p", type=str, help="Custom prompt for boundary detection")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        # Auto-generate output path
        args.output = args.input.parent / f"{args.input.stem}{config.OUTPUT_SUFFIX}.png"
    
    if args.input and not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.input_dir and not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Print header
    print("\n" + "="*60)
    print("  Field Boundary Detection - Gemini 3 Pro Image Preview")
    print("="*60)
    
    # Get API key securely
    api_key = get_api_key()
    
    # Initialize detector
    detector = FieldBoundaryDetector(
        api_key=api_key,
        resolution=args.resolution,
        prompt=args.prompt
    )
    
    # Process
    if args.input:
        print(f"\nProcessing: {args.input}")
        print(f"Output: {args.output}")
        
        if detector.process_image(args.input, args.output):
            print(f"\n✓ Success! Output saved to: {args.output}")
        else:
            print(f"\n✗ Failed to process image")
            sys.exit(1)
    else:
        detector.process_batch(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
