"""
Integrated Field Boundary Detection Pipeline
Combines Gemini AI + OpenCV to create georeferenced shapefiles

Just run: python3 integrated_field_detector.py
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
import geopandas as gpd
from shapely.geometry import Polygon
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

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

def pixel_to_latlon(x, y, center_lat, center_lon, zoom, W, H):
    """
    Convert pixel coordinates to lat/lon.
    
    Args:
        x, y: Pixel coordinates in image
        center_lat, center_lon: Geographic center of the image
        zoom: Map zoom level
        W, H: Image width and height
    
    Returns:
        (lat, lon) tuple
    """
    cpx = lon_to_px(center_lon, zoom)
    cpy = lat_to_px(center_lat, zoom)
    global_px = cpx + (x - W/2)
    global_py = cpy + (y - H/2)
    lat = px_to_lat(global_py, zoom)
    lon = px_to_lon(global_px, zoom)
    return lat, lon


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

def deg_per_pixel(lat_deg, zoom):
    """
    Compute degrees of latitude & longitude per pixel for a Google static map.
    """
    lat_rad = math.radians(lat_deg)
    scale = 2 ** zoom

    # Google Web Mercator pixel size formulas
    lon_deg_per_px = 360.0 / (256 * scale)
    lat_deg_per_px  = (360.0 / (256 * scale)) * math.cos(lat_rad)

    return lat_deg_per_px, lon_deg_per_px




def map_bounds(center_lat, center_lon, zoom, width, height):
    """
    Returns lat/lon of the 4 map corners.
    """
    lat_px, lon_px = deg_per_pixel(center_lat, zoom)

    half_w = width  / 2
    half_h = height / 2

    # North/South = latitude
    north = center_lat + half_h * lat_px
    south = center_lat - half_h * lat_px

    # East/West = longitude
    east  = center_lon + half_w * lon_px
    west  = center_lon - half_w * lon_px

    return north, south, east, west




def pixel_to_latlon_linear(x, y, north, south, east, west, W, H):
    """
    Convert pixel coordinate to lat/lon based ONLY on the 
    real-world 4-corner positions (correct method for Google Static Maps).
    """
    # y = 0 is TOP
    lat = north - (y / H) * (north - south)
    
    # x = 0 is LEFT
    lon = west + (x / W) * (east - west)

    return float(lat), float(lon)

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
# OPENCV CONTOUR EXTRACTION
# ============================================================================
# -------------------------
# Add this helper near top
# -------------------------
from PIL import Image as PILImage

def scale_contours_to_original(contours, processed_size: Tuple[int,int], original_size: Tuple[int,int]):
    """
    Scale contour point coordinates from the processed (boundary) image
    back to the original static map image pixel coordinates.

    contours: list of numpy arrays (n,1,2) as returned by cv2.findContours
    processed_size: (H_proc, W_proc) e.g. (2048, 2048)
    original_size: (H_orig, W_orig) e.g. (640, 640)

    Returns: list of scaled contours as numpy arrays of shape (n,2)
    """
    Hp, Wp = processed_size
    Ho, Wo = original_size

    sx = Wo / float(Wp)
    sy = Ho / float(Hp)

    scaled = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(float)
        pts[:, 0] = pts[:, 0] * sx  # x
        pts[:, 1] = pts[:, 1] * sy  # y
        scaled.append(pts.astype(int))
    return scaled



def extract_contours_from_boundaries(boundary_image_path: Path, output_dir: Path) -> List[np.ndarray]:
    """
    Extract field contours from Gemini's boundary-highlighted image.
    
    Returns:
        List of contours (each is numpy array of points)
    """
    print(f"\n{'='*70}")
    print("STEP 2/3: EXTRACTING FIELD CONTOURS")
    print('='*70)
    
    # Load the boundary image
    img = cv2.imread(str(boundary_image_path))
    if img is None:
        raise ValueError(f"Could not load: {boundary_image_path}")
    
    H, W = img.shape[:2]
    print(f"📸 Processing {W}x{H} image")
    
    # Convert to multiple color spaces for better detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # METHOD 1: Yellow color detection (original method)
    print("🎨 Method 1: Detecting yellow boundaries...")
    lower_yellow = np.array([15, 80, 80])  # Broader range
    upper_yellow = np.array([40, 255, 255])
    yellow_mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # METHOD 2: Detect orange/red boundaries too (some might be orange)
    print("🎨 Method 2: Detecting orange boundaries...")
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # METHOD 3: Edge detection on the boundary lines
    print("🎨 Method 3: Edge detection...")
    edges = cv2.Canny(gray, 50, 150)
    
    # Combine all masks
    print("🔄 Combining detection methods...")
    combined_mask = cv2.bitwise_or(yellow_mask1, orange_mask)
    combined_mask = cv2.bitwise_or(combined_mask, edges)
    
    # Enhance the mask with morphological operations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Close gaps in boundaries
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    # Dilate slightly to ensure boundaries are connected
    combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
    
    # Invert the mask - we want the areas INSIDE the boundaries
    print("🔄 Inverting mask to get field regions...")
    inverted_mask = cv2.bitwise_not(combined_mask)
    
    # Remove small noise
    inverted_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Fill small holes inside fields
    inverted_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Save debug images
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "1_yellow_mask.jpg"), yellow_mask1)
    cv2.imwrite(str(debug_dir / "2_orange_mask.jpg"), orange_mask)
    cv2.imwrite(str(debug_dir / "3_edges.jpg"), edges)
    cv2.imwrite(str(debug_dir / "4_combined_mask.jpg"), combined_mask)
    cv2.imwrite(str(debug_dir / "5_inverted_mask.jpg"), inverted_mask)
    
    # Find contours in the inverted mask (these are the field regions)
    print("🔍 Finding field contours...")
    contours, hierarchy = cv2.findContours(inverted_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours more intelligently
    min_area = 50  # Much smaller minimum - to catch small fields
    max_area = H * W * 0.3  # Maximum 30% of image
    
    print(f"📊 Total contours found: {len(contours)}")
    
    valid_contours = []
    filtered_stats = {"too_small": 0, "too_large": 0, "valid": 0}
    
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter by size
        if area < min_area:
            filtered_stats["too_small"] += 1
            continue
        if area > max_area:
            filtered_stats["too_large"] += 1
            continue
        
        # Check if it's an outer contour (not a hole)
        if hierarchy is not None:
            # hierarchy format: [Next, Previous, First_Child, Parent]
            # We want outer contours (Parent == -1) or first-level children
            if hierarchy[0][idx][3] != -1:  # Has a parent
                parent_area = cv2.contourArea(contours[hierarchy[0][idx][3]])
                # Skip if it's a small hole inside a larger field
                if area < parent_area * 0.1:
                    continue
        
        # Simplify contour to reduce points
        epsilon = 0.001 * cv2.arcLength(cnt, True)  # More detailed
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Must have at least 3 points to be a valid polygon
        if len(approx) >= 3:
            valid_contours.append(approx)
            filtered_stats["valid"] += 1
    
    print(f"📊 Filtering results:")
    print(f"   • Too small (< {min_area} px²): {filtered_stats['too_small']}")
    print(f"   • Too large (> {max_area} px²): {filtered_stats['too_large']}")
    print(f"   • Valid fields: {filtered_stats['valid']}")
    print(f"✅ Found {len(valid_contours)} valid field boundaries")
    
    # Create detailed visualization
    vis_img = img.copy()
    
    # Draw each contour with a different color for visibility
    for idx, cnt in enumerate(valid_contours):
        color = (0, 255, 0) if idx % 2 == 0 else (255, 0, 0)
        cv2.drawContours(vis_img, [cnt], -1, color, 2)
        
        # Add field number at centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(vis_img, str(idx+1), (cx-10, cy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    vis_path = output_dir / f"{boundary_image_path.stem}_contours.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"✅ Contour visualization saved: {vis_path}")
    
    return valid_contours


# ============================================================================
# SHAPEFILE GENERATION
# ============================================================================

# -------------------------
# DROP-IN REPLACEMENT
# -------------------------
def create_georeferenced_shapefiles(
    contours: List[np.ndarray],
    processed_image_shape: Tuple[int, int],   # shape of Gemini boundary image -> (H_proc, W_proc)
    output_dir: Path,
    output_name: str
) -> Tuple[int, int]:
    """
    Correct georeferencing: scale contours back to original static map size,
    then convert to lon/lat using Web-Mercator math at the configured zoom.
    """

    print(f"\n{'='*70}")
    print("STEP 3/3: CORRECTED GEOREFERENCING + SHAPEFILES (SCALED)")
    print('='*70)

    # 1) Get original static map pixel size (the one you actually requested from Google)
    orig_img = PILImage.open(Path(config.INPUT_IMAGE))
    Wo, Ho = orig_img.size          # PIL returns (width, height)
    # We'll use (H, W) convention elsewhere to match your code
    original_shape = (Ho, Wo)

    # processed_image_shape comes from cv2.imread(boundary_image).shape[:2] => (H_proc, W_proc)
    processed_shape = processed_image_shape

    print(f"📌 Original static map pixels (W x H): {Wo} x {Ho}")
    print(f"📌 Processed (boundary) image pixels (W x H): {processed_shape[1]} x {processed_shape[0]}")

    # 2) SCALE contours from processed -> original pixel grid
    scaled_contours_pts = scale_contours_to_original(contours, processed_shape, original_shape)
    print(f"🔁 Scaled {len(scaled_contours_pts)} contours back to original pixel grid")

    # 3) Map pixel -> global Web-Mercator pixel -> lat/lon using existing utilities
    H_orig, W_orig = original_shape
    center_lat = config.CENTER_LAT
    center_lon = config.CENTER_LON
    zoom = config.ZOOM_LEVEL

    # helper: world size at zoom in pixels
    ws = world_size(zoom)

    # center global pixel coords (floating)
    cpx = lon_to_px(center_lon, zoom)
    cpy = lat_to_px(center_lat, zoom)

    polygons = []
    skipped = 0

    for pts in scaled_contours_pts:
        if pts.shape[0] < 3:
            skipped += 1
            continue

        geo_coords = []
        # pts are (N,2): [x, y] where x is horizontal pixel (0..W_orig-1), y is vertical (0..H_orig-1)
        for x_px, y_px in pts:
            # Convert image pixel -> global pixel coordinates
            # Image center corresponds to (cpx, cpy). The offset in pixels is simply (x - W_orig/2)
            global_px = cpx + (float(x_px) - (W_orig / 2.0))
            global_py = cpy + (float(y_px) - (H_orig / 2.0))

            # Convert global pixel -> lon/lat (Web Mercator inverse)
            lon = px_to_lon(global_px, zoom)
            lat = px_to_lat(global_py, zoom)

            geo_coords.append((lon, lat))

        # close polygon
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

    # 4) Save shapefiles / geojson as EPSG:4326 (lon,lat)
    shapefile_dir = output_dir / "shapefiles"
    shapefile_dir.mkdir(parents=True, exist_ok=True)

    # Combined GeoDataFrame
    combined_gdf = gpd.GeoDataFrame(
        {
            'field_id': range(1, len(polygons) + 1),
            'area_deg2': [p.area for p in polygons]   # area in degrees (rough) — optional
        },
        geometry=polygons,
        crs="EPSG:4326"
    )

    combined_path = shapefile_dir / f"{output_name}_all_fields.shp"
    combined_gdf.to_file(combined_path)
    print(f"✅ Saved combined shapefile: {combined_path}")

    geojson_path = shapefile_dir / f"{output_name}_all_fields.geojson"
    combined_gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"✅ Also saved GeoJSON: {geojson_path}")

    # OPTIONAL: save individual shapefiles if you want
    individual_dir = shapefile_dir / "individual_fields"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for idx, poly in enumerate(polygons, start=1):
        gdf = gpd.GeoDataFrame({'field_id':[idx]}, geometry=[poly], crs="EPSG:4326")
        gdf.to_file(individual_dir / f"field_{idx:03d}.shp")

    print(f"✅ Saved {len(polygons)} individual shapefiles in: {individual_dir}")

    return len(polygons), 1



# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline - run everything."""
    print("\n" + "="*70)
    print("🌾 INTEGRATED FIELD BOUNDARY DETECTION PIPELINE")
    print("   Gemini AI + OpenCV → Georeferenced Shapefiles")
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
    print(f"   Center coordinates: ({config.CENTER_LAT}, {config.CENTER_LON})")
    print(f"   Zoom level: {config.ZOOM_LEVEL}")
    print(f"   Output directory: {config.OUTPUT_DIR}/")
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get API key
    api_key = get_api_key()
    
    try:
        # STEP 1: Gemini AI boundary detection
        boundary_image = detect_boundaries_with_gemini(input_path, api_key, output_dir)
        
        # STEP 2: Extract contours from boundary image
        contours = extract_contours_from_boundaries(boundary_image, output_dir)
        
        if len(contours) == 0:
            print("\n❌ No contours found. Try adjusting the detection parameters.")
            sys.exit(1)
        
        # STEP 3: Create georeferenced shapefiles
        num_individual, num_combined = create_georeferenced_shapefiles(
            contours,
            cv2.imread(str(boundary_image)).shape[:2],
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
        print(f"\n📁 OUTPUT FILES (in '{output_dir}/' folder):")
        print(f"   1️⃣  {input_path.stem}_gemini_boundaries.png  (AI-detected boundaries)")
        print(f"   2️⃣  {input_path.stem}_gemini_boundaries_contours.jpg  (Extracted contours)")
        print(f"   3️⃣  shapefiles/individual_fields/  (Individual field shapefiles)")
        print(f"   4️⃣  shapefiles/{input_path.stem}_all_fields.shp  (Combined shapefile)")
        print(f"   5️⃣  shapefiles/{input_path.stem}_all_fields.geojson  (GeoJSON format)")
        print(f"   6️⃣  debug/  (Debug images)")
        print("\n" + "="*70)
        print("🎉 All done! Check the shapefiles/ folder for outputs!")
        print("="*70 + "\n")
        
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
