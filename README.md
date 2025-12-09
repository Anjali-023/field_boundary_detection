# 🌾Field Boundary Detection

**Automated field boundary detection using Google Gemini AI and SAM**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

---

## 📌 What This Does

1. **Downloads** satellite imagery from Google Maps
2. **Detects** field boundaries using Gemini AI
3. **Segments** individual fields with SAM
4. **Outputs** georeferenced shapefiles (`.shp`, `.geojson`)
5. **Evaluates** accuracy against ground truth using IoU metrics

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install google-genai opencv-python numpy pillow geopandas shapely matplotlib contextily pandas tqdm python-dotenv
```

### 2. Get API Keys

- **Google AI API**: https://aistudio.google.com/apikey
- **Google Maps API**: https://console.cloud.google.com/

Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_key_here
```

### 3. Download Satellite Image

```bash
# Visit this URL (replace YOUR_KEY):
https://maps.googleapis.com/maps/api/staticmap?center=28.72,78.55&zoom=16&size=2048x2048&maptype=satellite&key=YOUR_KEY

# Save as: input/satellite.jpg
```

### 4. Configure

Edit `config.py`:
```python
INPUT_IMAGE = "input/satellite.jpg"
CENTER_LAT = 28.7043
CENTER_LON = 78.5228
ZOOM_LEVEL = 16
OUTPUT_DIR = "output"
```

### 5. Run Pipeline

```bash
# Full pipeline
python integrated_field_detector.py

# Or individual steps:
python gemini_detector.py              # Step 1: Detect boundaries
python sam_segmentation.py              # Step 2: Extract fields
python evaluation.py                    # Step 3: Calculate IoU
```

---

## 📁 Project Structure

```
agricultural-field-detection/
├── config.py                          # Configuration (Document #1)
├── gemini_detector.py                 # Gemini AI detection (Document #2)
├── integrated_field_detector.py       # Full pipeline (Document #3)
├── evaluation.py                      # IoU evaluation (Document #4)
├── requirements.txt
├── .env                               # API keys
├── .gitignore
├── README.md
├── input/
│   └── satellite.jpg                  # Your satellite image
└── output/
    ├── boundaries.png                 # AI-detected boundaries
    ├── shapefiles/
    │   ├── field_001.shp
    │   ├── field_002.shp
    │   └── all_fields.geojson
    └── visualizations/
        ├── overall_comparison.png
        ├── zoom_regions.png
        └── iou_distribution.png
```

---

## 📝 File Mapping

| Your Document | Create This File | What It Does |
|--------------|------------------|--------------|
| Document #1 | `config.py` | All settings (coordinates, zoom, paths) |
| Document #2 | `gemini_detector.py` | Gemini AI boundary detection |
| Document #3 | `integrated_field_detector.py` | Complete pipeline (Gemini + SAM + Shapefiles) |
| Document #4 | `evaluation.py` | IoU calculation & visualizations |

---

## ⚙️ Configuration Options

**Zoom Levels:**
- `14` = Large regions (~10 km²)
- `16` = **Recommended** (~2.5 km²)
- `18` = Small fields (~600 m²)

**Output Resolution:**
- `1K` = Fast (1024×1024)
- `2K` = **Recommended** (2048×2048)
- `4K` = Best quality (4096×4096)

---

## 📊 Output Files

After running, check `output/` folder:

```
output/
├── satellite_gemini_boundaries.png       # Yellow boundaries overlay
├── satellite_gemini_boundaries_contours.jpg  # Detected field outlines
├── shapefiles/
│   ├── individual_fields/
│   │   ├── field_001.shp                # Individual shapefiles
│   │   ├── field_002.shp
│   │   └── ...
│   ├── satellite_all_fields.shp         # Combined shapefile
│   └── satellite_all_fields.geojson     # GeoJSON format
└── visualizations/
    ├── 01_overall_comparison.png        # Full area comparison
    ├── 02_zoom_regions.png              # Detail views
    ├── 03_best_worst_matches.png        # Quality analysis
    └── 04_iou_distribution.png          # Accuracy histogram
```

---

## 🧪 Evaluation Metrics

The system calculates:
- **IoU** (Intersection over Union) per field
- **Mean IoU** across all fields
- **Detection rate** (fields found vs ground truth)

Sample output:
```
📊 Overall Metrics:
  • Total Fields: 45
  • Mean IoU: 0.734
  • Median IoU: 0.758
  • Fields with IoU > 0.5: 84.4%
```

---

## 🛠️ Troubleshooting

**No fields detected?**
- Check if boundaries are yellow in `boundaries.png`
- Adjust `YELLOW_HUE_MIN/MAX` in config
- Try different `ZOOM_LEVEL` (16-17 works best)

**Coordinates wrong?**
- Verify `CENTER_LAT`, `CENTER_LON` match your image
- Check zoom level matches downloaded image

**API errors?**
- Verify API keys in `.env` file
- Check API quotas (Gemini: 50/day free)


---

## 🙏 Acknowledgment

- Google Gemini AI for boundary detection

---

## 📧 Support

- Open an issue for bugs
- Star ⭐ if this helps your research!

---

**Made for agricultural technology research** 🌾
