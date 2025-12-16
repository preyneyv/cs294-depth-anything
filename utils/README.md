# Utility Scripts

Diagnostic and utility scripts for the depth estimation pipeline.

## Scripts

### `check_image_dimensions.py`
Checks if image dimensions match camera calibration (K matrix).
Useful for identifying resolution mismatches that cause undistortion failures.

**Usage:**
```bash
python utils/check_image_dimensions.py \
    --rosbag dpt_rosbag_lvms_2024_12/rosbag2_2024_12_12-18_21_55_12.mcap \
    --image-topic /vimba_front/image \
    --camera-info-topic /vimba_front/camera_info
```

### `inspect_rosbag.py`
Inspects ROS2 MCAP bag contents - lists all topics, message types, and sample data.

**Usage:**
```bash
# List all topics
python utils/inspect_rosbag.py

# Check specific topic
python utils/inspect_rosbag.py --topic /vimba_front/image_raw

# Sample messages
python utils/inspect_rosbag.py --topic /vimba_front/image --sample
```

