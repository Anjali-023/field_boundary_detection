"""
Integrated Field Boundary Detection Pipeline with SAM
Combines Gemini AI + SAM (Segment Anything Model) to create georeferenced shapefiles

Installation:
pip install segment-anything torch torchvision opencv-python numpy pillow geopandas shapely tqdm python-dotenv google-genai

Download SAM checkpoint:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Just run: python3 sam_field_detector.py
All settings are in config.py
"""

import os
import sys
import math
import getpass
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import Polygon
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# SAM imports
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("❌ Missing SAM dependency")
    print("Install: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ Missing Google AI dependency")
    print("Install: pip install google-genai")
    sys.exit(1)

import config


# ============================================================================
# COORDINATE CONVERSION FUNCTIONS
# ============================================================================

def world_size(zoom):
    """Get world size in pixels at given zoom level."""
    return 256 * (2 ** zoom)

def lat_to_mercator(lat):
    """Convert latitude to Mercator projection."""
    return math.log(math.tan(math.radians(lat)/2 + math.pi/4))

def mercator_to_lat(m):
    """Convert Mercator projection to latitude."""
    return math.degrees(2 * math.atan(math.exp(m)) - math.pi/2)

def lon_to_px(lon, zoom):
    """Convert longitude to global pixel x-coordinate."""
    return (lon + 180) / 360 * world_size(zoom)

def lat_to_px(lat, zoom):
    """Convert latitude to global pixel y-coordinate."""
    m = lat_to_mercator(lat)
    return (1 - m/math.pi) / 2 * world_size(zoom)

def px_to_lon(px, zoom):
    """Convert global pixel x-coordinate to longitude."""
    return px / world_size(zoom) * 360 - 180

def px_to_lat(py, zoom):
    """Convert global pixel y-coordinate to latitude."""
    m = math.pi * (1 - 2 * py / world_size(zoom))
    return mercator_to_lat(m)


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_key() -> str:
    """Get API key securely from environment, .env, or interactive prompt."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key and api_key != "your_api_key_here":
        print("✓ API key loaded from environment")
        return api_key
    
    print("\n" + "="*70)
    print("🔑 Google AI API Key Required")
    print("="*70)
    print("Get your key: https://aistudio.google.com/apikey")
    print("Your input will NOT be displayed as you type.")
    print("-"*70)
    
    api_key = getpass.getpass("Enter your Google AI API key: ")
    
    if not api_key:
        print("❌ No API key provided")
        sys.exit(1)
    
    save = input("\n💾 Save to .env file for future use? (y/n): ").lower().strip()
    if save == 'y':
        env_path = Path(__file__).parent / ".env"
        with open(env_path, 'w') as f:
            f.write(f"GOOGLE_API_KEY={api_key}\n")
        print(f"✓ Saved to {env_path}")
    
    return api_key


# ============================================================================
# GEMINI AI BOUNDARY DETECTION
# ============================================================================

def detect_boundaries_with_gemini(image_path: Path, api_key: str, output_dir: Path) -> Path:
    """
    Use Gemini AI to detect and highlight field boundaries.
    
    Returns:
        Path to the boundary-highlighted image
    """
    print(f"\n{'='*70}")
    print("STEP 1/3: DETECTING BOUNDARIES WITH GEMINI AI")
    print('='*70)
    
    client = genai.Client(api_key=api_key)
    
    # Load image and determine aspect ratio
    image = Image.open(image_path)
    width, height = image.size
    ratio = width / height
    
    ratios = {
        "1:1": 1.0, "2:3": 2/3, "3:2": 3/2, "3:4": 3/4, "4:3": 4/3,
        "4:5": 4/5, "5:4": 5/4, "9:16": 9/16, "16:9": 16/9, "21:9": 21/9,
    }
    aspect_ratio = min(ratios.items(), key=lambda x: abs(x[1] - ratio))[0]
    
    print(f"📸 Image size: {width}x{height}")
    print(f"📐 Aspect ratio: {aspect_ratio}")
    print(f"🤖 Using model: {config.MODEL_NAME}")
    print(f"🎯 Resolution: {config.RESOLUTION}")
    
    # Call Gemini API with retry logic
    print("🌐 Calling Gemini API...")
    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=config.MODEL_NAME,
                contents=[config.DEFAULT_PROMPT, image],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=config.RESOLUTION
                    ),
                )
            )
            break
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                print(f"⚠️  Retry {attempt + 1}/{config.MAX_RETRIES}: {e}")
                time.sleep(config.RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(f"Gemini API failed after {config.MAX_RETRIES} attempts: {e}")
    
    # Save the boundary-highlighted image
    boundary_path = None
    for part in response.parts:
        if part.inline_data is not None:
            result_image = part.as_image()
            boundary_path = output_dir / f"{image_path.stem}_gemini_boundaries.png"
            result_image.save(boundary_path)
            print(f"✅ Boundary image saved: {boundary_path}")
            break
    
    if boundary_path is None:
        raise RuntimeError("Gemini did not return an image")
    
    return boundary_path


# ============================================================================
# SAM SEGMENTATION
# ============================================================================

def load_sam_model(checkpoint_path: str = "sam_vit_h_4b8939.pth"):
    """Load SAM model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading SAM model on {device.upper()}...")
    
    if not Path(checkpoint_path).exists():
        print(f"\n❌ SAM checkpoint not found: {checkpoint_path}")
        print("\n📥 Download SAM checkpoint:")
        print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("\nOr use a smaller model:")
        print("   sam_vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
        print("   sam_vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        sys.exit(1)
    
    # Determine model type from checkpoint name
    if "vit_h" in checkpoint_path:
        model_type = "vit_h"
    elif "vit_l" in checkpoint_path:
        model_type = "vit_l"
    elif "vit_b" in checkpoint_path:
        model_type = "vit_b"
    else:
        model_type = "vit_h"  # default
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    return sam, device


def segment_fields_with_sam(
    original_image_path: Path,
    boundary_image_path: Path,
    output_dir: Path,
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
) -> List[np.ndarray]:
    """
    Use SAM to segment individual fields from the boundary-highlighted image.
    
    Returns:
        List of contours (each is numpy array of points)
    """
    print(f"\n{'='*70}")
    print("STEP 2/3: SEGMENTING FIELDS WITH SAM")
    print('='*70)
    
    # Load SAM model
    sam, device = load_sam_model(sam_checkpoint)
    
    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.SAM_POINTS_PER_SIDE,
        pred_iou_thresh=config.SAM_PRED_IOU_THRESH,
        stability_score_thresh=config.SAM_STABILITY_SCORE_THRESH,
        crop_n_layers=config.SAM_CROP_N_LAYERS,
        crop_n_points_downscale_factor=config.SAM_CROP_N_POINTS_DOWNSCALE_FACTOR,
        min_mask_region_area=config.SAM_MIN_MASK_REGION_AREA,
    )
    
    # Load the original image (for SAM processing)
    print("📸 Loading original image for SAM...")
    original_img = cv2.imread(str(original_image_path))
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    H_orig, W_orig = original_img.shape[:2]
    print(f"📸 Original image size: {W_orig}x{H_orig}")
    
    # Generate masks with SAM
    print("🤖 Running SAM segmentation (this may take a minute)...")
    masks = mask_generator.generate(original_img_rgb)
    print(f"✓ SAM generated {len(masks)} segments")
    
    # Filter and sort masks by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Convert masks to contours
    print("🔄 Converting SAM masks to contours...")
    contours = []
    valid_masks = []
    
    min_area = config.SAM_MIN_FIELD_AREA_PIXELS
    max_area = H_orig * W_orig * config.SAM_MAX_FIELD_AREA_RATIO
    
    for mask_dict in tqdm(masks, desc="Processing masks"):
        mask = mask_dict['segmentation']
        area = mask_dict['area']
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Find contours in the mask
        mask_uint8 = mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) == 0:
            continue
        
        # Take the largest contour (should be the field boundary)
        cnt = max(cnts, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = config.SAM_CONTOUR_EPSILON * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) >= 3:
            contours.append(approx)
            valid_masks.append(mask)
    
    print(f"✅ Filtered to {len(contours)} valid field contours")
    
    # Create visualization
    print("🎨 Creating visualization...")
    vis_img = original_img.copy()
    
    # Draw all contours
    for idx, cnt in enumerate(contours):
        # Alternate colors for visibility
        color = (0, 255, 0) if idx % 2 == 0 else (255, 0, 0)
        cv2.drawContours(vis_img, [cnt], -1, color, 2)
        
        # Add field number at centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(vis_img, str(idx+1), (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    vis_path = output_dir / f"{original_image_path.stem}_sam_contours.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"✅ Visualization saved: {vis_path}")
    
    # Save mask overlay
    mask_overlay = original_img.copy()
    for idx, mask in enumerate(valid_masks):
        color = np.random.randint(0, 255, 3).tolist()
        mask_overlay[mask] = (np.array(mask_overlay[mask]) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    
    mask_path = output_dir / f"{original_image_path.stem}_sam_masks.jpg"
    cv2.imwrite(str(mask_path), mask_overlay)
    print(f"✅ Mask overlay saved: {mask_path}")
    
    # Save debug info
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    
    return contours


# ============================================================================
# SHAPEFILE GENERATION
# ============================================================================

def scale_contours_to_original(contours, processed_size: Tuple[int,int], original_size: Tuple[int,int]):
    """Scale contour coordinates if needed (for this version, sizes should match)."""
    Hp, Wp = processed_size
    Ho, Wo = original_size
    
    if (Hp, Wp) == (Ho, Wo):
        # No scaling needed
        return [cnt.reshape(-1, 2) for cnt in contours]
    
    sx = Wo / float(Wp)
    sy = Ho / float(Hp)
    
    scaled = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(float)
        pts[:, 0] = pts[:, 0] * sx
        pts[:, 1] = pts[:, 1] * sy
        scaled.append(pts.astype(int))
    return scaled


def create_georeferenced_shapefiles(
    contours: List[np.ndarray],
    processed_image_shape: Tuple[int, int],
    output_dir: Path,
    output_name: str
) -> Tuple[int, int]:
    """Create georeferenced shapefiles from contours."""
    
    print(f"\n{'='*70}")
    print("STEP 3/3: CREATING GEOREFERENCED SHAPEFILES")
    print('='*70)
    
    # Get original image dimensions
    orig_img = Image.open(Path(config.INPUT_IMAGE))
    Wo, Ho = orig_img.size
    original_shape = (Ho, Wo)
    
    print(f"📌 Original image pixels (W x H): {Wo} x {Ho}")
    
    # Scale contours if needed
    scaled_contours_pts = scale_contours_to_original(contours, processed_image_shape, original_shape)
    
    # Get map parameters
    H_orig, W_orig = original_shape
    center_lat = config.CENTER_LAT
    center_lon = config.CENTER_LON
    zoom = config.ZOOM_LEVEL
    
    # Convert contours to geographic coordinates
    cpx = lon_to_px(center_lon, zoom)
    cpy = lat_to_px(center_lat, zoom)
    
    polygons = []
    skipped = 0
    
    print("🌍 Converting to geographic coordinates...")
    for pts in tqdm(scaled_contours_pts, desc="Processing fields"):
        if pts.shape[0] < 3:
            skipped += 1
            continue
        
        geo_coords = []
        for x_px, y_px in pts:
            # Convert to global pixel coordinates
            global_px = cpx + (float(x_px) - (W_orig / 2.0))
            global_py = cpy + (float(y_px) - (H_orig / 2.0))
            
            # Convert to lon/lat
            lon = px_to_lon(global_px, zoom)
            lat = px_to_lat(global_py, zoom)
            
            geo_coords.append((lon, lat))
        
        # Close polygon
        if geo_coords[0] != geo_coords[-1]:
            geo_coords.append(geo_coords[0])
        
        try:
            poly = Polygon(geo_coords)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
            else:
                skipped += 1
        except Exception as e:
            print(f"⚠️  Skipped invalid polygon: {e}")
            skipped += 1
    
    print(f"✓ Created {len(polygons)} valid georeferenced polygons (skipped {skipped})")
    
    if len(polygons) == 0:
        print("❌ No valid polygons created")
        return 0, 0
    
    # Save shapefiles
    shapefile_dir = output_dir / "shapefiles"
    shapefile_dir.mkdir(parents=True, exist_ok=True)
    
    # Combined shapefile
    combined_gdf = gpd.GeoDataFrame(
        {
            'field_id': range(1, len(polygons) + 1),
            'area_deg2': [p.area for p in polygons]
        },
        geometry=polygons,
        crs="EPSG:4326"
    )
    
    combined_path = shapefile_dir / f"{output_name}_all_fields.shp"
    combined_gdf.to_file(combined_path)
    print(f"✅ Saved combined shapefile: {combined_path}")
    
    geojson_path = shapefile_dir / f"{output_name}_all_fields.geojson"
    combined_gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"✅ Saved GeoJSON: {geojson_path}")
    
    # Individual shapefiles
    if config.SAVE_INDIVIDUAL_SHAPEFILES:
        individual_dir = shapefile_dir / "individual_fields"
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, poly in enumerate(polygons, start=1):
            gdf = gpd.GeoDataFrame({'field_id':[idx]}, geometry=[poly], crs="EPSG:4326")
            gdf.to_file(individual_dir / f"field_{idx:03d}.shp")
        
        print(f"✅ Saved {len(polygons)} individual shapefiles")
    
    return len(polygons), 1


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline - run everything."""
    print("\n" + "="*70)
    print("🌾 SAM-BASED FIELD BOUNDARY DETECTION PIPELINE")
    print("   Gemini AI + SAM → Georeferenced Shapefiles")
    print("="*70)
    
    # Validate config
    input_path = Path(config.INPUT_IMAGE)
    if not input_path.exists():
        print(f"\n❌ ERROR: Input image not found!")
        print(f"   Looking for: {input_path}")
        print(f"\n💡 Fix: Edit config.py and set INPUT_IMAGE to your image file")
        sys.exit(1)
    
    print(f"\n📋 Configuration:")
    print(f"   Input image: {input_path}")
    print(f"   Center: ({config.CENTER_LAT}, {config.CENTER_LON})")
    print(f"   Zoom: {config.ZOOM_LEVEL}")
    print(f"   SAM checkpoint: {config.SAM_CHECKPOINT}")
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API key
    api_key = get_api_key()
    
    try:
        # STEP 1: Gemini AI boundary detection (optional - can skip if you want)
        if config.USE_GEMINI_FOR_PREPROCESSING:
            boundary_image = detect_boundaries_with_gemini(input_path, api_key, output_dir)
        else:
            boundary_image = input_path
            print("\n⏭️  Skipping Gemini preprocessing, using original image")
        
        # STEP 2: SAM segmentation
        contours = segment_fields_with_sam(
            input_path,
            boundary_image,
            output_dir,
            config.SAM_CHECKPOINT
        )
        
        if len(contours) == 0:
            print("\n❌ No fields detected. Try adjusting SAM parameters.")
            sys.exit(1)
        
        # STEP 3: Create georeferenced shapefiles
        num_individual, num_combined = create_georeferenced_shapefiles(
            contours,
            cv2.imread(str(input_path)).shape[:2],
            output_dir,
            input_path.stem
        )
        
        # Final summary
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n📊 RESULTS:")
        print(f"   • Fields detected: {num_individual}")
        print(f"   • Individual shapefiles: {num_individual}")
        print(f"   • Combined shapefile: {num_combined}")
        print(f"\n📁 OUTPUT FILES:")
        print(f"   1️⃣  {input_path.stem}_sam_contours.jpg (Detected boundaries)")
        print(f"   2️⃣  {input_path.stem}_sam_masks.jpg (Segmentation masks)")
        print(f"   3️⃣  shapefiles/{input_path.stem}_all_fields.shp (Combined)")
        print(f"   4️⃣  shapefiles/{input_path.stem}_all_fields.geojson (GeoJSON)")
        print(f"   5️⃣  shapefiles/individual_fields/ (Individual fields)")
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
