# File Organization Plan

## Current Main Directory Files

### ✅ KEEP IN MAIN (Core/Production)
- `pointcloudpipeline/` - Main pipeline code (keep as-is)
- `proj/` - Project code (keep as-is)
- `dataset/` - Data output (keep as-is)
- `dpt_rosbag_lvms_2024_12/` - Rosbag data and extraction (keep as-is)
- `Depth-Anything-V2/` - External dependency (keep as-is)
- `README.md` - Main documentation (keep)

### 🔧 MOVE TO `utils/` (Diagnostic/Utility Scripts)
- `check_image_dimensions.py` - Useful diagnostic tool
- `inspect_rosbag.py` - Useful diagnostic tool

### 📓 KEEP IN MAIN (Notebooks - but could organize)
- `clouds.ipynb` - Development notebook
- `metric_demo.ipynb` - Demo notebook
- `test.ipynb` - Test notebook

### 🗑️ DELETE (Test/Temporary Files)
- `frame_000.png` through `frame_004.png` - Test frames (move to test_data or delete)

## Proposed Structure

```
depth-estimation/
├── README.md
├── pointcloudpipeline/          # Main pipeline (keep)
├── proj/                         # Project code (keep)
├── dataset/                      # Data output (keep)
├── dpt_rosbag_lvms_2024_12/     # Rosbag data (keep)
├── Depth-Anything-V2/            # External dependency (keep)
├── utils/                        # NEW: Utility scripts
│   ├── check_image_dimensions.py
│   └── inspect_rosbag.py
├── notebooks/                    # NEW: Optional - organize notebooks
│   ├── clouds.ipynb
│   ├── metric_demo.ipynb
│   └── test.ipynb
└── test_data/                    # NEW: Test files
    └── frames/                   # Move frame_*.png here or delete
```

## Action Plan

1. **Create `utils/` folder** - Move diagnostic scripts here
2. **Delete test frames** - Or move to test_data if needed
3. **Optional: Create `notebooks/` folder** - If you want to organize notebooks

