"""
Configuration file for Integrated Field Boundary Detection
Edit these settings before running the script
"""

# ============================================================================
# INPUT IMAGE SETTINGS - EDIT THESE!
# ============================================================================

# Your satellite image file (in the same directory as the script)
INPUT_IMAGE = ""

# Geographic coordinates of the image center
CENTER_LAT = 28.7043      #Your AOI coordinates
CENTER_LON = 78.5228      
ZOOM_LEVEL = 16           # Map zoom level (typically 14-18 for fields)

# Output directory
OUTPUT_DIR = ""


# ============================================================================
# GEMINI AI SETTINGS
# ============================================================================

# Gemini model that supports image generation
MODEL_NAME = "gemini-3-pro-image-preview"

# Output resolution: "1K", "2K", or "4K"
# Higher = better quality but slower processing
RESOLUTION = "2K"

# API retry settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


# ============================================================================
# DETECTION PROMPT (Advanced users only)
# ============================================================================

DEFAULT_PROMPT = """You are an expert agricultural land surveyor and remote sensing specialist analyzing a satellite image.

TASK: Analyze this satellite imagery and generate a new image with all agricultural field boundaries precisely traced and highlighted.

DETECTION INSTRUCTIONS:
1. IDENTIFY all individual agricultural fields, farmlands, crop parcels, and cultivated areas
2. DETECT boundaries between different fields based on:
   - Visible crop lines and field edges
   - Color/texture differences between adjacent plots
   - Irrigation channels and pathways between fields
   - Fencing lines and property demarcations
   - Roads, paths, and access routes that separate fields
   - Natural boundaries like streams, hedgerows, or tree lines

DRAWING SPECIFICATIONS:
- Color: Bright yellow (#FFFF00) for maximum visibility
- Line thickness: 2-3 pixels - visible but not obscuring field details
- Line style: Solid, continuous lines following the exact boundary contours
- Ensure lines are crisp and well-defined, not blurry or feathered

OUTPUT REQUIREMENTS:
1. PRESERVE the original satellite image exactly as the background
2. OVERLAY only the yellow boundary lines on top
3. DO NOT modify, enhance, or change the underlying satellite imagery
4. DO NOT add any text, labels, legends, or annotations
5. Trace EVERY visible agricultural field boundary in the entire image
6. Ensure boundaries form closed polygons where fields are fully visible
7. For partially visible fields at image edges, trace the visible portions

ACCURACY PRIORITIES:
- Precision over speed - trace boundaries exactly along visible edges
- Include small fields and irregular shaped parcels
- Distinguish between individual plots even if crops appear similar
- Follow natural contours and irregular field shapes accurately

Generate the output image now with all field boundaries highlighted in yellow."""


# ============================================================================
# OPENCV DETECTION SETTINGS (Advanced users only)
# ============================================================================

# Yellow color detection range in HSV
YELLOW_HUE_MIN = 15        # Lower threshold to catch more yellows
YELLOW_HUE_MAX = 40        # Higher threshold
YELLOW_SAT_MIN = 80        # Lower saturation threshold
YELLOW_VAL_MIN = 80        # Lower value threshold

# Orange/Red detection (for boundaries that appear orange)
ORANGE_HUE_MIN = 5
ORANGE_HUE_MAX = 20

# Contour filtering
MIN_FIELD_AREA_PIXELS = 50        # Smaller minimum to catch small fields
MAX_FIELD_AREA_RATIO = 0.3        # Maximum field size as ratio of image

# Contour simplification (0.001-0.01, lower = more detail)
CONTOUR_EPSILON = 0.001           # More detailed boundaries


# ============================================================================
# SHAPEFILE SETTINGS
# ============================================================================

# Coordinate Reference System
CRS = "EPSG:4326"  # WGS84 - standard lat/lon

# Output formats
SAVE_INDIVIDUAL_SHAPEFILES = True
SAVE_COMBINED_SHAPEFILE = True
SAVE_GEOJSON = True


# ============================================================================
# NOTES
# ============================================================================

"""
ZOOM LEVEL GUIDE:
- 14: ~10 km² visible, use for very large fields
- 15: ~5 km² visible, good for large agricultural regions
- 16: ~2.5 km² visible, recommended for most use cases ✓
- 17: ~1.2 km² visible, good for detailed field mapping
- 18: ~600 m² visible, use for small/irregular fields

HOW TO GET YOUR COORDINATES:
1. Go to Google Maps
2. Find your area of interest
3. Right-click on the center of the image
4. Select the coordinates (they'll be copied)
5. First number is latitude, second is longitude

RESOLUTION GUIDE:
- 1K: Fast processing, lower quality (1024x1024)
- 2K: Balanced speed/quality (2048x2048) ✓ Recommended
- 4K: Best quality, slower (4096x4096)
"""
