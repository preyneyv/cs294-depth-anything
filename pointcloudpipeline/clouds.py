"""
Pointcloud Pipeline

A pipeline that extracts and processes Luminar LiDAR point clouds from ROS/MCAP bags.
Based on the clouds.ipynb notebook.

This module provides functionality to:
- Parse Luminar PointCloud2 messages into structured NumPy arrays
- Compute ray identifiers for point matching between frames
- Interpolate point clouds between frames for temporal alignment
- Visualize point clouds interactively
- Filter and process point cloud data
"""

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# ROS bag type store initialization (used for deserializing ROS messages)
typestore = get_typestore(Stores.LATEST)

# Constants
TWO_PI = 2.0 * np.pi  # Full circle in radians, used for azimuth wrapping


def luminar_cloud_to_struct(msg):
    """
    Parse a Luminar-style PointCloud2 ROS message into a structured NumPy array.

    This function extracts all point data from a Luminar LiDAR PointCloud2 message
    and converts it into a structured NumPy array with named fields. The function
    handles the specific binary layout of Luminar point cloud data, including
    spatial coordinates, timestamps, sensor metadata, and quality metrics.

    Args:
        msg: A ROS sensor_msgs/PointCloud2 message from a Luminar LiDAR sensor.
             Must have attributes: point_step, width, height, and data.

    Returns:
        np.ndarray: A structured array with dtype containing the following fields:
            - timestamp (uint64): Per-point timestamp in nanoseconds
            - x, y, z (float32): 3D Cartesian coordinates in meters
            - reflectance (float32): Reflectance/intensity value
            - return_index (uint8): Which return this is (1st, 2nd, etc.)
            - last_return_index (uint8): Total number of returns for this ray
            - sensor_id (uint8): Identifier of the sensor that captured this point
            - azimuth (float32): Horizontal angle in radians
            - elevation (float32): Vertical angle in radians
            - depth (float32): Radial distance in meters
            - line_index (uint16): Scan line identifier
            - frame_index (uint8): Frame sequence number
            - detector_site_id (uint8): Detector site identifier
            - scan_checkpoint (uint8): Scan checkpoint marker
            - existence_prob (uint8): Existence probability percentage (0-100)
            - data_qualifier (uint8): Data quality flag
            - blockage_level (uint8): Blockage/occlusion level

    Note:
        The function does not filter points by default. Points with existence_prob == 0
        or other invalid data may still be present in the output array.
    """
    # Get the size of each point in bytes and total number of points
    point_step = msg.point_step
    n_points = msg.width * msg.height

    # Define the structured dtype matching Luminar's binary point format
    # This matches the exact byte layout of the PointCloud2 message
    dtype = np.dtype({
        'names': [
            'timestamp', 'x', 'y', 'z', 'reflectance',
            'return_index', 'last_return_index', 'sensor_id',
            'azimuth', 'elevation', 'depth',
            'line_index', 'frame_index', 'detector_site_id',
            'scan_checkpoint', 'existence_prob',
            'data_qualifier', 'blockage_level'
        ],
        'formats': [
            '<u8',   # timestamp (8 bytes, little-endian uint64)
            '<f4',   # x
            '<f4',   # y
            '<f4',   # z
            '<f4',   # reflectance
            'u1',    # return_index
            'u1',    # last_return_index
            'u1',    # sensor_id
            '<f4',   # azimuth
            '<f4',   # elevation
            '<f4',   # depth
            '<u2',   # line_index
            'u1',    # frame_index
            'u1',    # detector_site_id
            'u1',    # scan_checkpoint
            'u1',    # existence_probability_percent
            'u1',    # data_qualifier
            'u1',    # blockage_level
        ],
        'offsets': [
            0,   # timestamp
            8,   # x
            12,  # y
            16,  # z
            20,  # reflectance
            24,  # return_index
            25,  # last_return_index
            26,  # sensor_id
            32,  # azimuth
            36,  # elevation
            40,  # depth
            44,  # line_index
            46,  # frame_index
            47,  # detector_site_id
            48,  # scan_checkpoint
            49,  # existence_probability_percent
            50,  # data_qualifier
            51,  # blockage_level
        ],
        'itemsize': point_step,  # Total size of each point in bytes
    })

    # Parse the binary message data into the structured array
    # This creates a view (not a copy) of the data, so it's memory efficient
    arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)

    # Note: Optionally filter obvious junk here (e.g., existence_prob == 0)
    # Currently returns all points without filtering
    return arr


def compute_ray_id(arr, azimuth_bins=36000):
    """
    Compute a stable ray identifier from line_index and azimuth angle.

    This function creates a unique identifier for each LiDAR ray by combining
    the scan line index and quantized azimuth angle. The ray ID is used to
    match corresponding points between different point cloud frames, enabling
    temporal interpolation and tracking.

    The ray ID is a 64-bit integer where:
    - Upper 32 bits: line_index (which scan line the point belongs to)
    - Lower 32 bits: Quantized azimuth bin index (horizontal angle discretized)

    Args:
        arr: Structured NumPy array from luminar_cloud_to_struct() containing
             'line_index' and 'azimuth' fields.
        azimuth_bins (int, optional): Number of bins to quantize azimuth over
             the full 2π range. Default is 36000, giving 0.01° resolution.
             Higher values give finer angular resolution but may cause hash
             collisions if line_index values are large.

    Returns:
        np.ndarray: Array of int64 ray identifiers, one per point in arr.
                    Each ID uniquely identifies a ray based on its scan line
                    and horizontal angle.

    Example:
        If a point has line_index=5 and azimuth=1.57 radians (90°), and
        azimuth_bins=36000, the azimuth bin would be ~9000, and the ray_id
        would be (5 << 32) | 9000 = a large unique integer.
    """
    # Extract azimuth angles from the structured array
    az = arr['azimuth']
    
    # Wrap azimuth to [0, 2π) range to handle any values outside this range
    # This ensures all angles are normalized before quantization
    az_wrapped = np.mod(az, TWO_PI)

    # Quantize azimuth to a bin index
    # Divide the full circle into azimuth_bins equal segments
    # Floor operation ensures we get integer bin indices
    az_idx = np.floor(az_wrapped * (azimuth_bins / TWO_PI)).astype(np.int64)
    
    # Clip to valid range [0, azimuth_bins-1] to handle edge cases
    # (e.g., if azimuth is exactly 2π after wrapping)
    az_idx = np.clip(az_idx, 0, azimuth_bins - 1)

    # Extract line_index and convert to int64 for bit operations
    # line_index is originally uint16, so we have plenty of room in upper 32 bits
    line_idx = arr['line_index'].astype(np.int64)

    # Pack line_index and az_idx into one 64-bit integer
    # Strategy: [line_index (upper 32 bits) | az_idx (lower 32 bits)]
    # Shift line_index left by 32 bits, then OR with az_idx
    # This gives us a unique identifier for each unique (line, azimuth) pair
    ray_id = (line_idx << 32) | az_idx

    return ray_id


def interpolate_luminar_frames(arr0, arr1, t=0.5, azimuth_bins=36000):
    """
    Interpolate between two Luminar PointCloud2 frames at a specified time fraction.

    This function matches corresponding points between two point cloud frames
    using ray identifiers, then performs linear interpolation of their positions,
    reflectance values, and timestamps. This is useful for temporal alignment
    when you need a point cloud at a time between two captured frames.

    The matching process:
    1. Computes ray IDs for both frames using line_index and azimuth
    2. Packs ray_id with return_index to create unique point keys
    3. Finds intersection of keys to identify matching points
    4. Linearly interpolates matched points based on time fraction t

    Args:
        arr0: Structured NumPy array from luminar_cloud_to_struct() for the
              earlier frame (at time t=0).
        arr1: Structured NumPy array from luminar_cloud_to_struct() for the
              later frame (at time t=1).
        t (float, optional): Interpolation fraction in [0, 1]. 
             - t=0.0 returns points from arr0
             - t=0.5 returns points at the midpoint
             - t=1.0 returns points from arr1
             Default is 0.5 (midpoint).
        azimuth_bins (int, optional): Number of azimuth bins for ray ID computation.
             Must match the value used in compute_ray_id(). Default is 36000.

    Returns:
        tuple: A 5-tuple containing:
            - xyz_mid (np.ndarray): Shape (M, 3) float32 array of interpolated
              XYZ coordinates, where M is the number of matched points.
            - reflectance_mid (np.ndarray): Shape (M,) float32 array of
              interpolated reflectance values.
            - timestamp_mid (np.ndarray): Shape (M,) uint64 array of interpolated
              timestamps in nanoseconds.
            - meta0_idx (np.ndarray): Shape (M,) int64 array of indices into arr0
              for the matched points.
            - meta1_idx (np.ndarray): Shape (M,) int64 array of indices into arr1
              for the matched points.

    Note:
        - Returns empty arrays if either input is empty or no matches are found.
        - Only points that appear in both frames (matched by ray_id + return_index)
          are included in the output.
        - The interpolation is linear in both space and time.
    """
    # Handle empty input arrays
    if len(arr0) == 0 or len(arr1) == 0:
        # Return empty arrays with correct dtypes if either frame is empty
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.uint64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64))

    # --- Build ray IDs for both frames ---
    # Compute unique ray identifiers for each point in both frames
    # This identifies which LiDAR ray each point came from
    ray0 = compute_ray_id(arr0, azimuth_bins=azimuth_bins)
    ray1 = compute_ray_id(arr1, azimuth_bins=azimuth_bins)

    # --- Create composite keys for point matching ---
    # Pack (ray_id, return_index) into one key to uniquely identify each point
    # return_index distinguishes multiple returns from the same ray (e.g., 1st vs 2nd return)
    # Shift ray_id left by 3 bits (room for return_index values 0-7) and OR with return_index
    key0 = (ray0 << 3) | arr0['return_index'].astype(np.int64)
    key1 = (ray1 << 3) | arr1['return_index'].astype(np.int64)

    # --- Find matching points between frames ---
    # intersect1d finds common keys and returns:
    # - common_keys: the keys that appear in both arrays
    # - idx0: indices in arr0 where these keys appear
    # - idx1: indices in arr1 where these keys appear
    # assume_unique=False allows duplicate keys (though they should be unique per frame)
    common_keys, idx0, idx1 = np.intersect1d(key0, key1, assume_unique=False,
                                             return_indices=True)

    # Handle case where no points match between frames
    if common_keys.size == 0:
        # Return empty arrays but preserve the index arrays (which will also be empty)
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.uint64),
                idx0, idx1)

    # Extract the matched points from both frames
    # These are the points that correspond to each other across frames
    A0 = arr0[idx0]  # Matched points from frame 0
    A1 = arr1[idx1]  # Matched points from frame 1

    # --- Extract fields to interpolate ---
    # Get XYZ coordinates from matched points in both frames
    x0 = A0['x'].astype(np.float32)
    y0 = A0['y'].astype(np.float32)
    z0 = A0['z'].astype(np.float32)

    x1 = A1['x'].astype(np.float32)
    y1 = A1['y'].astype(np.float32)
    z1 = A1['z'].astype(np.float32)

    # Get reflectance values for interpolation
    refl0 = A0['reflectance'].astype(np.float32)
    refl1 = A1['reflectance'].astype(np.float32)

    # Get timestamps for temporal interpolation
    ts0 = A0['timestamp'].astype(np.uint64)
    ts1 = A1['timestamp'].astype(np.uint64)

    # --- Perform linear interpolation ---
    # Convert t to float to ensure proper arithmetic
    t = float(t)

    t = max(0.0, min(1.0, float(t))) # ensure t is between 0 and 1
    
    # Interpolate XYZ coordinates: (1-t)*point0 + t*point1
    # Stack into (M, 3) array where each row is [x, y, z]
    xyz_mid = np.stack([
        (1.0 - t) * x0 + t * x1,  # Interpolated X coordinates
        (1.0 - t) * y0 + t * y1,  # Interpolated Y coordinates
        (1.0 - t) * z0 + t * z1,  # Interpolated Z coordinates
    ], axis=-1).astype(np.float32)

    # Interpolate reflectance values
    reflectance_mid = ((1.0 - t) * refl0 + t * refl1).astype(np.float32)

    # Interpolate timestamps
    # Convert to float64 for precision, then back to uint64
    # This handles the large timestamp values correctly
    ts_mid = ((1.0 - t) * ts0.astype(np.float64) + t * ts1.astype(np.float64)).astype(np.uint64)

    return xyz_mid, reflectance_mid, ts_mid, idx0, idx1


def show_cloud(*xyzs, max_points=200_000):
    """
    Create an interactive 3D scatter plot of one or more point clouds using Plotly.

    This function visualizes point clouds in a web-based interactive 3D viewer.
    If point clouds are too large, they are automatically downsampled to maintain
    performance. Multiple point clouds can be displayed simultaneously, each as
    a separate trace in the plot.

    Args:
        *xyzs: Variable number of point cloud arrays. Each should be a NumPy array
               of shape (N, 3) where N is the number of points and columns are
               [x, y, z] coordinates.
        max_points (int, optional): Maximum number of points to display per cloud.
               If a cloud has more points, it will be uniformly downsampled.
               Default is 200,000 points per cloud.

    Returns:
        None: Displays the plot in a browser window (blocking call).

    Raises:
        ValueError: If any input point cloud is empty (has zero points).

    Example:
        >>> cloud1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        >>> cloud2 = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 3]])
        >>> show_cloud(cloud1, cloud2)  # Displays both clouds in one plot
    """
    data = []
    
    # Process each point cloud provided as input
    for xyz in xyzs:
        # Validate that the point cloud is not empty
        if len(xyz) == 0:
            raise ValueError("Empty point cloud")

        # Downsample for performance if the cloud is too large
        # This prevents browser crashes and slow rendering with millions of points
        if len(xyz) > max_points:
            # Calculate step size for uniform downsampling
            # e.g., if we have 500k points and max_points=200k, step=2 (take every 2nd point)
            step = len(xyz) // max_points
            xyz = xyz[::step]  # Slice with step to downsample
        
        # Create a Plotly 3D scatter trace for this point cloud
        data.append(
            go.Scatter3d(
                x=xyz[:, 0],  # X coordinates
                y=xyz[:, 1],  # Y coordinates
                z=xyz[:, 2],  # Z coordinates
                mode="markers",  # Display as point cloud (not lines)
                marker=dict(
                    size=1,      # Point size in pixels
                    opacity=0.5, # Semi-transparent for better visualization
                ),
            )
        )

    # Create the figure with all point cloud traces
    fig = go.Figure(data=data)

    # Configure the 3D scene layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",      # Label for X axis
            yaxis_title="Y",      # Label for Y axis
            zaxis_title="Z",      # Label for Z axis
            aspectmode="data",    # Preserve scale (1:1:1) so distances are accurate
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins for full-screen view
    )

    # Display the interactive plot (opens in browser)
    fig.show()


def make_xyz(arr):
    """
    Extract XYZ coordinates from a structured point cloud array and apply spatial filters.

    This function converts a structured NumPy array (from luminar_cloud_to_struct) into
    a simple (N, 3) array of XYZ coordinates, while applying basic spatial filtering
    to remove points that are likely noise or outside the region of interest.

    The current filters remove:
    - Points with y < -20 (behind the sensor or too far back)
    - Points with x > 50 (too far to the right/side)

    Args:
        arr: Structured NumPy array from luminar_cloud_to_struct() containing
             'x', 'y', 'z' fields.

    Returns:
        np.ndarray: Shape (M, 3) float32 array of filtered XYZ coordinates,
                  where M <= N (number of points after filtering).
                  Each row is [x, y, z] in meters.

    Note:
        The filter thresholds (-20 for y, 50 for x) are hardcoded and may need
        adjustment based on your specific use case and sensor mounting position.
    """
    # Extract XYZ coordinates from the structured array
    # Stack x, y, z fields into a (N, 3) array
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=-1).astype(np.float32)
    
    # Create a boolean mask for filtering points
    # Filter out points with y < -20 (behind sensor or too far back)
    mask = xyz[:, 1] > -20.0
    
    # Further filter: remove points with x > 50 (too far to the side)
    # Using &= to combine with previous mask (both conditions must be true)
    mask &= xyz[:, 0] < 50.0
    
    # Return only the points that pass both filters
    return xyz[mask]

def load_luminar_clouds(bag_path: Path, topic: str = "/luminar_front/points/existence_prob_filtered"):
    """
    Load all Luminar point cloud frames from a ROS2/MCAP bag.

    Args:
        bag_path: Path to the MCAP or ROS bag file. Can be relative or absolute.
        topic: ROS topic name to extract point clouds from. Default is the Luminar
               front sensor topic with existence probability filtering.

    Returns:
        List of (timestamp_ns: int, points: np.ndarray) tuples, sorted by time.

    Raises:
        FileNotFoundError: If the bag file doesn't exist.
        ValueError: If the file format cannot be determined or is unsupported.
    """
    # Resolve the path to absolute to avoid relative path issues
    bag_path = bag_path.resolve()
    
    # Verify the file exists
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    
    # Verify it's a file (not a directory)
    if not bag_path.is_file():
        raise ValueError(f"Path is not a file: {bag_path}")
    
    clouds: list[tuple[int, np.ndarray]] = []

    # Detect file format by checking the file header
    # MCAP files start with: 0x89 0x4D 0x43 0x41 0x50 (MCAP in ASCII)
    # Format: \x89MCAP<version><newline> where version is typically '0' or '1'
    mcap_magic = b'\x89MCAP'
    with open(bag_path, 'rb') as f:
        file_header = f.read(8)  # Read first 8 bytes to see full header
    is_mcap = file_header[:5] == mcap_magic  # Check first 5 bytes for MCAP magic
    
    if is_mcap:
        print(f"Detected MCAP format (header: {[hex(b) for b in file_header]})")
    else:
        print(f"File format detection: header = {[hex(b) for b in file_header[:6]]}")

    # AnyReader should auto-detect MCAP vs ROS bag format
    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            # Find all connections matching the desired topic
            connections = [c for c in reader.connections if c.topic == topic]
            
            if not connections:
                raise ValueError(f"Topic '{topic}' not found in bag file. "
                               f"Available topics: {[c.topic for c in reader.connections]}")
            
            # Read all messages from this topic
            for conn, timestamp, raw_msg in reader.messages(connections=connections):
                # Deserialize the ROS message
                pc = reader.deserialize(raw_msg, conn.msgtype)
                
                # Convert to structured NumPy array
                arr = luminar_cloud_to_struct(pc)
                
                # Store timestamp and point cloud array
                clouds.append((int(timestamp), arr))
    
    except Exception as e:
        # Provide helpful error message based on detected file format
        if is_mcap:
            raise ValueError(
                f"Failed to read MCAP file '{bag_path}'. "
                f"The file is confirmed to be MCAP format (header: {[hex(b) for b in file_header[:5]]}). "
                f"Ensure rosbags>=0.11.0 and mcap-ros2-support are installed. "
                f"Error: {e}"
            ) from e
        else:
            raise ValueError(
                f"Failed to read bag file '{bag_path}'. "
                f"Expected MCAP or ROS bag format. Error: {e}"
            ) from e

    # Sort by timestamp to ensure chronological order
    clouds.sort(key=lambda x: x[0])
    return clouds

def cloud_indices(clouds):
    """
    Returns the timestamps of the cloud frames.
    """
    return np.array([ts for ts, _ in clouds], dtype=np.int64)

def load_camera():
    """
    Loads all camera frames from a ROS bag. Saves each one in a separate image file. Returns 
    a list of timestamps. 

    also writes to a index file.
    """

def find_surrounding_cloud_indices(timestamp, cloud_timestamps):
    """
    Finds the nearest 2 cloud indices to the given timestamp.
    """
    i = np.searchsorted(cloud_timestamps, timestamp)
    if i == 0:
        return 0, 0
    if i >= len(cloud_timestamps):
        return len(cloud_timestamps) - 1, len(cloud_timestamps) - 1
    return i - 1, i

def calculate_camera_to_lidar_transform():
    """
    Inputs: 
    - list of timestamps from the camera
    - list of timestamps from the lidar
    Outputs:
    - for each camera frame, the nearest 2 lidar point clouds
    - scalar value needed to interpolate the camera frame to the nearest 2 lidar point cloud
    """

def iteratively_call_luminar_interpolation():
    """
    this could just be main:

    """

def main():
    """
    Main function: Extract point clouds from ROS bag and demonstrate interpolation.

    This function:
    1. Opens a ROS/MCAP bag file
    2. Extracts all Luminar front point cloud messages
    3. Converts them to structured arrays
    4. Demonstrates interpolation between two frames
    5. Visualizes the original and interpolated point clouds

    Note:
        The bag file path is hardcoded. Modify the Path() argument to point
        to your specific bag file location.
    """
    # Extract all point clouds from rosbag and convert to xyz numpy arrays
    # Open the MCAP/ROS bag file for reading
    # Using resolve() to convert relative path to absolute based on script location
    script_dir = Path(__file__).parent
    bag_path = script_dir / "../dpt_rosbag_lvms_2024_12/rosbag2_2024_12_12-18_21_55_12.mcap"
    
    print(f"Loading point clouds from: {bag_path.resolve()}")
    clouds = load_luminar_clouds(bag_path)
    print(f"Loaded {len(clouds)} point cloud frames")

    # Extract two consecutive frames for interpolation demonstration
    # clouds[1] and clouds[2] are the second and third point clouds in the sequence
    c1 = clouds[1][1]  # Structured array from frame 1
    c2 = clouds[2][1]  # Structured array from frame 2

    # Interpolate between the two frames at the midpoint (t=0.5)
    # This creates a synthetic point cloud at a time halfway between the two captures
    xyz_mid, refl_mid, ts_mid, idx0, idx1 = interpolate_luminar_frames(
        c1, c2,
        t=0.5,           # Interpolate at the midpoint
        azimuth_bins=36000,  # Use default azimuth binning resolution
    )
    
    # Visualize the original two frames and the interpolated result
    # make_xyz() extracts and filters the XYZ coordinates from each structured array
    show_cloud(make_xyz(c1), make_xyz(c2), xyz_mid)




if __name__ == "__main__":
    main()
