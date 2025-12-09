import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box
from shapely.ops import unary_union
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import contextily as ctx

ESRI_SAT = {
    "url": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "attribution": "Esri World Imagery"
}

# ======================
# PATHS
# ======================
sam_shapefile_folder = ""
farm_csv_path = "GROUND_TRUTH_PATH"
output_folder = ""

import os
os.makedirs(output_folder, exist_ok=True)

# ======================
# HELPER FUNCTION: ADD ACCURACY BOUNDARY
# ======================
def add_accuracy_boundary(ax, bounds, margin_percent=0.15, style='hatched'):
    """
    Add visual indicators showing edge regions are less accurate
    
    Parameters:
    - ax: matplotlib axis
    - bounds: (xmin, ymin, xmax, ymax) of the total area
    - margin_percent: percentage of area to mark as edge zone (0.15 = 15%)
    - style: 'hatched', 'shaded', or 'both'
    """
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    
    margin_x = width * margin_percent
    margin_y = height * margin_percent
    
    # Define edge zones (rectangles)
    edge_zones = [
        # Left edge
        mpatches.Rectangle((xmin, ymin), margin_x, height, 
                          linewidth=0, edgecolor='none', facecolor='red', alpha=0.15),
        # Right edge
        mpatches.Rectangle((xmax - margin_x, ymin), margin_x, height,
                          linewidth=0, edgecolor='none', facecolor='red', alpha=0.15),
        # Top edge (excluding corners already covered)
        mpatches.Rectangle((xmin + margin_x, ymax - margin_y), width - 2*margin_x, margin_y,
                          linewidth=0, edgecolor='none', facecolor='red', alpha=0.15),
        # Bottom edge (excluding corners already covered)
        mpatches.Rectangle((xmin + margin_x, ymin), width - 2*margin_x, margin_y,
                          linewidth=0, edgecolor='none', facecolor='red', alpha=0.15),
    ]
    
    # Add shaded regions if requested
    if style in ['shaded', 'both']:
        for zone in edge_zones:
            ax.add_patch(zone)
    
    # Add hatched borders if requested
    if style in ['hatched', 'both']:
        # Outer boundary (full area)
        outer_rect = mpatches.Rectangle((xmin, ymin), width, height,
                                       linewidth=3, edgecolor='red', 
                                       facecolor='none', linestyle='--', alpha=0.8)
        ax.add_patch(outer_rect)
        
        # Inner boundary (accurate center region)
        inner_rect = mpatches.Rectangle((xmin + margin_x, ymin + margin_y), 
                                       width - 2*margin_x, height - 2*margin_y,
                                       linewidth=3, edgecolor='green', 
                                       facecolor='none', linestyle='-', alpha=0.8)
        ax.add_patch(inner_rect)
        
    # Add text labels
    ax.text(xmin + margin_x/2, (ymin + ymax)/2, 'EDGE\nZONE', 
           rotation=90, ha='center', va='center', 
           fontsize=10, fontweight='bold', color='red', alpha=0.7,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.text((xmin + xmax)/2, (ymin + ymax)/2, 'HIGH ACCURACY\nCENTER REGION', 
           ha='center', va='center', 
           fontsize=12, fontweight='bold', color='green', alpha=0.8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ======================
# 1. LOAD DATA
# ======================
print("Loading data...")
df_farm = pd.read_csv(farm_csv_path)
wkt_col = "WKT" if "WKT" in df_farm.columns else "wkt"
df_farm = df_farm.dropna(subset=[wkt_col])
df_farm["geometry"] = df_farm[wkt_col].apply(wkt.loads)
gdf_farm = gpd.GeoDataFrame(df_farm, geometry="geometry", crs="EPSG:4326")

shp_files = glob.glob(f"{sam_shapefile_folder}/*.shp")
gdf_sam = gpd.GeoDataFrame(
    pd.concat([gpd.read_file(shp) for shp in shp_files], ignore_index=True)
)
if gdf_sam.crs is None:
    gdf_sam = gdf_sam.set_crs("EPSG:4326")

# Project to meters
gdf_farm = gdf_farm.to_crs("EPSG:3857")
gdf_sam = gdf_sam.to_crs("EPSG:3857")

# Filter manual fields in SAM area
sam_bbox = gdf_sam.total_bounds
sam_area = box(*sam_bbox)
gdf_farm_filtered = gdf_farm[gdf_farm.geometry.intersects(sam_area)].copy()

print(f"Manual fields: {len(gdf_farm_filtered)}")
print(f"SAM segments: {len(gdf_sam)}")

# ======================
# CLEAN + STRAIGHTEN SAM FIELD GEOMETRIES
# ======================
from shapely.validation import make_valid
from shapely.geometry import Polygon

print("\nConverting SAM polygons to proper rectangles...")

def rect_polygon(poly):
    if poly.is_empty:
        return poly
    poly = make_valid(poly)
    rect = poly.minimum_rotated_rectangle
    if not rect.is_valid:
        rect = rect.buffer(0)
    return rect

gdf_sam["geometry"] = gdf_sam["geometry"].apply(rect_polygon)
print("✓ Polygon rectangulation complete!")

# ======================
# 2. VISUALIZE OVERALL OVERLAP WITH BOUNDARIES
# ======================
print("\n[1/3] Creating overlap visualization with accuracy boundaries...")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for ax, title, plot_func in zip(
    axes,
    ["Manual Ground Truth Fields", "SAM Segmented Fields", "Both Overlapped"],
    [lambda a: gdf_farm_filtered.plot(ax=a, facecolor='none', edgecolor='lime', linewidth=2),
     lambda a: gdf_sam.plot(ax=a, facecolor='none', edgecolor='yellow', linewidth=1),
     lambda a: (gdf_farm_filtered.plot(ax=a, facecolor='none', edgecolor='lime', linewidth=2),
                gdf_sam.plot(ax=a, facecolor='none', edgecolor='yellow', linewidth=1))]
):
    plot_func(ax)
    ctx.add_basemap(ax, crs="EPSG:3857", source=ESRI_SAT["url"], zoom=17)
    
    # Add accuracy boundary indicator
    add_accuracy_boundary(ax, sam_bbox, margin_percent=0.15, style='both')
    
    ax.set_title(title, fontsize=14, fontweight="bold", color='white')
    ax.grid(False)

plt.tight_layout()
plt.savefig(f"{output_folder}/01_overall_with_basemap.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: 01_overall_with_basemap.png")

# ======================
# 3. ZOOM INTO PROBLEM AREAS WITH BOUNDARIES
# ======================
print("\n[2/3] Creating zoomed problem area views...")

centroids = gdf_farm_filtered.geometry.centroid
x_coords = centroids.x.values
y_coords = centroids.y.values

x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

zoom_regions = [
    (x_min, x_min + (x_max-x_min)/3, y_min, y_min + (y_max-y_min)/3),
    (x_min + (x_max-x_min)/3, x_min + 2*(x_max-x_min)/3, y_min, y_min + (y_max-y_min)/3),
    (x_min, x_min + (x_max-x_min)/3, y_min + (y_max-y_min)/3, y_min + 2*(y_max-y_min)/3),
    (x_min + (x_max-x_min)/3, x_min + 2*(x_max-x_min)/3, y_min + (y_max-y_min)/3, y_min + 2*(y_max-y_min)/3),
]

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

for idx, (x1, x2, y1, y2) in enumerate(zoom_regions):
    ax = axes[idx]
    
    zoom_box = box(x1, y1, x2, y2)
    farm_zoom = gdf_farm_filtered[gdf_farm_filtered.geometry.intersects(zoom_box)]
    sam_zoom = gdf_sam[gdf_sam.geometry.intersects(zoom_box)]
    
    if len(farm_zoom) > 0:
        farm_zoom.plot(ax=ax, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2, alpha=0.4, label='Manual')
        sam_zoom.plot(ax=ax, facecolor='lightcoral', edgecolor='darkred', linewidth=1.5, alpha=0.4, label='SAM')
        
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        
        # Add boundary indicator for this zoom region
        add_accuracy_boundary(ax, (x1, y1, x2, y2), margin_percent=0.12, style='hatched')
        
        ax.set_title(f'Zoom Region {idx+1}\n({len(farm_zoom)} manual, {len(sam_zoom)} SAM)', 
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No fields in region', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Zoom Region {idx+1} (Empty)', fontsize=12)

plt.tight_layout()
plt.savefig(f"{output_folder}/02_zoom_regions.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_folder}/02_zoom_regions.png")
plt.close()

# ======================
# 4. SHOW BEST AND WORST MATCHES
# ======================
print("\n[3/3] Creating best/worst match examples with satellite basemap...")

results = []
for farm_idx, farm_row in gdf_farm_filtered.iterrows():
    farm_poly = farm_row.geometry
    best_iou = 0
    best_sam_poly = None

    for _, sam_row in gdf_sam.iterrows():
        sam_geom = sam_row.geometry
        if not sam_geom.intersects(farm_poly):
            continue

        inter = farm_poly.intersection(sam_geom).area
        union = farm_poly.union(sam_geom).area
        iou = inter / union if union > 0 else 0

        if iou > best_iou:
            best_iou = iou
            best_sam_poly = sam_geom

    results.append({
        "farm_poly": farm_poly,
        "sam_poly": best_sam_poly,
        "iou": best_iou
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("iou", ascending=False)

valid_results = results_df[results_df["iou"] > 0.01]
valid_results = valid_results.sort_values("iou", ascending=False)

best_matches = valid_results.head(6)
worst_matches = valid_results.tail(6)

fig, axes = plt.subplots(3, 4, figsize=(22, 16))
axes = axes.flatten()

pairs = (
    [(f"Best Match #{i+1}", row, "green") for i, row in best_matches.iterrows()] +
    [(f"Worst Match #{i+1}", row, "red") for i, row in worst_matches.iterrows()]
)

for ax, (title, row, title_color) in zip(axes, pairs):
    farm_poly = row["farm_poly"]
    sam_poly = row["sam_poly"]

    if sam_poly is None:
        ax.text(0.5, 0.5, "No match", ha="center", va="center")
        continue

    gdf_f = gpd.GeoDataFrame([{"geometry": farm_poly}], crs="EPSG:3857")
    gdf_s = gpd.GeoDataFrame([{"geometry": sam_poly}], crs="EPSG:3857")

    gdf_f.plot(ax=ax, facecolor="none", edgecolor="lime", linewidth=2, label="Manual")
    gdf_s.plot(ax=ax, facecolor="none", edgecolor="yellow", linewidth=2, label="SAM")

    xmin, ymin, xmax, ymax = gdf_f.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery, zoom=18)

    ax.set_title(f"{title}\nIoU = {row['iou']:.3f}", color=title_color)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_folder}/03_best_worst_matches_satellite.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {output_folder}/03_best_worst_matches_satellite.png")

# ======================
# 5. IoU DISTRIBUTION
# ======================
print("\n[4/4] Creating IoU distribution histogram...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(results_df['iou'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(results_df['iou'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {results_df["iou"].mean():.3f}')
ax.axvline(results_df['iou'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {results_df["iou"].median():.3f}')
ax.set_xlabel('IoU Score', fontsize=12)
ax.set_ylabel('Number of Fields', fontsize=12)
ax.set_title('Distribution of IoU Scores\n(Manual vs SAM Segmentation)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{output_folder}/04_iou_distribution.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_folder}/04_iou_distribution.png")
plt.close()

print("\n" + "="*60)
print("✓ VISUALIZATION COMPLETE")
print("="*60)
print(f"\nAll images saved to: {output_folder}/")
print("\nFiles created:")
print("  1. 01_overall_comparison.png - Full area with accuracy boundaries")
print("  2. 02_zoom_regions.png - 4 zoomed regions with edge indicators")
print("  3. 03_best_worst_matches.png - Individual field comparisons")
print("  4. 04_iou_distribution.png - IoU histogram")
