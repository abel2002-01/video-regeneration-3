#!/usr/bin/env python3
"""
================================================================================
SCRIPT 2: BUILD_CORRESPONDENCE.PY - Correspondence Map Builder
================================================================================

Purpose:
    Build a mapping between surface points (vertices) and their pixel locations
    across all rotation frames. This enables tracking the same physical surface
    point as the object rotates.

The Core Problem:
    When the camera is fixed and only the object rotates, the same pixel
    sees DIFFERENT surface points in each frame. Without correspondence
    tracking, sampling a fixed pixel gives meaningless mixed data.

The Solution:
    1. Each vertex has a unique ID encoded as RGB color
    2. For each frame, we read which vertex is visible at each pixel
    3. We build a reverse mapping: vertex -> [(frame, y, x), ...]
    4. Now we can find where any surface point appears in any frame!

How Vertex ID Decoding Works:
    The RGB color encodes vertex index in base-256:
        vertex_index = R*255 + G*255*256 + B*255*65536
    
    This allows us to identify up to 16.7 million unique vertices.

Run:
    python build_correspondence.py

Input:
    renders/theta_XXX_vertexid.exr  - Vertex ID maps from render.py

Output:
    correspondence/correspondence_map.json - The vertex->pixel mapping
    correspondence/stats.json              - Statistics about the mapping
    logs/correspondence.log                - Execution log

Author: Abel & Team
Date: January 2026
================================================================================
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Try to import EXR reading library
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("WARNING: imageio not found. Install with: pip install imageio")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
RENDERS_DIR = SCRIPT_DIR / "renders"
CORRESPONDENCE_DIR = SCRIPT_DIR / "correspondence"
LOGS_DIR = SCRIPT_DIR / "logs"

# Visibility threshold: pixels with RGB sum above this are considered visible
VISIBILITY_THRESHOLD = 0.001

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class Logger:
    """
    Logging system that writes to both console and file.
    
    Provides detailed tracking of the correspondence building process,
    which is essential for debugging and verification.
    """
    
    def __init__(self, log_file: Path):
        """Initialize logger with output file path."""
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"CORRESPONDENCE LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def section(self, title: str):
        """Log a section header."""
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)
    
    def success(self, message: str):
        """Log success message."""
        self.log(message, "SUCCESS")
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")

# Initialize logger
logger = Logger(LOGS_DIR / "correspondence.log")

# ============================================================================
# EXR READING
# ============================================================================

def read_exr(filepath: Path) -> np.ndarray:
    """
    Read an EXR image file and return as numpy array.
    
    EXR Format:
        - 32-bit floating point per channel
        - Supports values outside [0, 1]
        - Lossless compression
    
    Args:
        filepath: Path to the EXR file
    
    Returns:
        numpy array of shape (height, width, channels)
    
    Raises:
        ImportError: If no EXR library is available
        FileNotFoundError: If the file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"EXR file not found: {filepath}")
    
    if HAS_IMAGEIO:
        # imageio.v2 to avoid deprecation warning
        import imageio.v2 as iio
        img = iio.imread(str(filepath))
        return img[:, :, :3] if img.shape[-1] > 3 else img
    else:
        raise ImportError("No EXR library available. Install imageio: pip install imageio")


def decode_vertex_id(rgb: np.ndarray) -> int:
    """
    Decode RGB color to vertex index.
    
    The Encoding Scheme:
        We use base-256 encoding where each color channel represents
        a "digit" in a base-256 number:
        
        vertex_index = R_int + G_int * 256 + B_int * 65536
        
        Where R_int, G_int, B_int are the 0-255 integer values.
    
    Why This Works:
        - 256^3 = 16,777,216 unique values
        - Covers any reasonable mesh vertex count
        - Simple encoding/decoding
    
    Args:
        rgb: Array of [R, G, B] values in range [0, 1]
    
    Returns:
        Integer vertex index
    """
    r_int = int(round(rgb[0] * 255))
    g_int = int(round(rgb[1] * 255))
    b_int = int(round(rgb[2] * 255))
    
    return r_int + g_int * 256 + b_int * 65536


def get_visibility_mask(vertex_id_map: np.ndarray) -> np.ndarray:
    """
    Create a mask of visible (non-background) pixels.
    
    Background pixels have RGB values very close to zero (transparent).
    We use a small threshold to identify visible object pixels.
    
    Args:
        vertex_id_map: The vertex ID image of shape (H, W, 3)
    
    Returns:
        Boolean mask of shape (H, W) where True = visible
    """
    # Sum of RGB channels - visible pixels have non-zero values
    rgb_sum = np.sum(np.abs(vertex_id_map), axis=-1)
    return rgb_sum > VISIBILITY_THRESHOLD


# ============================================================================
# CORRESPONDENCE BUILDING
# ============================================================================

def build_correspondence_map(renders_dir: Path, theta_samples: int) -> Dict[str, Any]:
    """
    Build the complete correspondence map from vertex ID renders.
    
    Algorithm:
        For each frame (rotation angle):
            For each visible pixel:
                1. Read the RGB color at that pixel
                2. Decode to vertex index
                3. Add to mapping: vertex_key -> [(frame, y, x), ...]
    
    The resulting map allows us to answer:
        "Where does vertex V appear across all frames?"
    
    Why vertex_key is a string:
        - JSON doesn't support tuple keys
        - We convert (R, G, B) tuple to string for storage
        - Rounding to 4 decimals handles floating-point imprecision
    
    Args:
        renders_dir: Directory containing vertex ID EXR files
        theta_samples: Number of rotation angles rendered
    
    Returns:
        Dictionary with:
            - vertex_to_pixels: {vertex_key: [[frame, y, x], ...]}
            - metadata: {theta_samples, resolution, ...}
    """
    logger.log(f"Building correspondence map from {theta_samples} frames...")
    
    vertex_to_pixels: Dict[str, List[List[int]]] = {}
    total_observations = 0
    
    for frame_idx in range(theta_samples):
        # Load vertex ID map for this frame
        vid_path = renders_dir / f"theta_{frame_idx:03d}_vertexid.exr"
        
        if not vid_path.exists():
            logger.warning(f"Missing vertex ID map: {vid_path}")
            continue
        
        vid_map = read_exr(vid_path)
        visibility_mask = get_visibility_mask(vid_map)
        
        # Count visible pixels
        visible_count = np.sum(visibility_mask)
        
        # Process each visible pixel
        visible_coords = np.argwhere(visibility_mask)
        
        for y, x in visible_coords:
            rgb = vid_map[y, x, :3]
            
            # Create vertex key from rounded RGB values
            # Rounding handles floating-point imprecision from EXR
            vertex_key = str(tuple(np.round(rgb, 4)))
            
            if vertex_key not in vertex_to_pixels:
                vertex_to_pixels[vertex_key] = []
            
            vertex_to_pixels[vertex_key].append([int(frame_idx), int(y), int(x)])
            total_observations += 1
        
        # Log progress every 10 frames
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            logger.log(f"  Frame {frame_idx + 1}/{theta_samples}: {visible_count} visible pixels")
    
    # Get resolution from first frame
    first_vid = read_exr(renders_dir / "theta_000_vertexid.exr")
    resolution = first_vid.shape[:2]
    
    # Build result
    result = {
        "vertex_to_pixels": vertex_to_pixels,
        "metadata": {
            "theta_samples": theta_samples,
            "resolution": list(resolution),
            "total_vertices": len(vertex_to_pixels),
            "total_observations": total_observations,
            "avg_visibility": total_observations / len(vertex_to_pixels) if vertex_to_pixels else 0
        }
    }
    
    return result


def compute_statistics(corr_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute detailed statistics about the correspondence map.
    
    These statistics help verify the quality of correspondence:
        - How many unique vertices were found?
        - How often is each vertex visible?
        - Are there vertices visible in all frames?
    
    Args:
        corr_map: The correspondence map from build_correspondence_map
    
    Returns:
        Dictionary of statistics
    """
    vertex_to_pixels = corr_map["vertex_to_pixels"]
    theta_samples = corr_map["metadata"]["theta_samples"]
    
    # Count visibility per vertex
    visibility_counts = [len(pixels) for pixels in vertex_to_pixels.values()]
    
    if not visibility_counts:
        return {"error": "No vertices found"}
    
    visibility_array = np.array(visibility_counts)
    
    stats = {
        "total_unique_vertices": len(vertex_to_pixels),
        "total_observations": sum(visibility_counts),
        "visibility": {
            "min": int(np.min(visibility_array)),
            "max": int(np.max(visibility_array)),
            "mean": float(np.mean(visibility_array)),
            "median": float(np.median(visibility_array)),
            "std": float(np.std(visibility_array))
        },
        "coverage": {
            "visible_in_all_frames": int(np.sum(visibility_array >= theta_samples)),
            "visible_in_50pct_frames": int(np.sum(visibility_array >= theta_samples * 0.5)),
            "visible_in_25pct_frames": int(np.sum(visibility_array >= theta_samples * 0.25)),
            "visible_in_10pct_frames": int(np.sum(visibility_array >= theta_samples * 0.1))
        },
        "theta_samples": theta_samples
    }
    
    return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function to build and save the correspondence map.
    
    Pipeline Steps:
        1. Load metadata from render.py
        2. Read all vertex ID maps
        3. Build vertex -> pixel mapping
        4. Compute statistics
        5. Save results
    """
    logger.section("CORRESPONDENCE MAP BUILDER")
    
    # Verify input directory exists
    if not RENDERS_DIR.exists():
        logger.error(f"Renders directory not found: {RENDERS_DIR}")
        logger.error("Please run render.py first!")
        return
    
    # Load metadata
    metadata_path = RENDERS_DIR / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        render_metadata = json.load(f)
    
    theta_samples = render_metadata["theta_samples"]
    resolution = render_metadata["resolution"]
    vertex_count = render_metadata["vertex_count"]
    
    logger.log(f"Resolution: {resolution}x{resolution}")
    logger.log(f"Theta samples: {theta_samples}")
    logger.log(f"Expected vertices: {vertex_count}")
    
    # Build correspondence map
    logger.section("BUILDING CORRESPONDENCE MAP")
    start_time = datetime.now()
    
    corr_map = build_correspondence_map(RENDERS_DIR, theta_samples)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.log(f"Correspondence built in {elapsed:.1f} seconds")
    
    # Compute statistics
    logger.section("COMPUTING STATISTICS")
    stats = compute_statistics(corr_map)
    
    logger.log(f"Total unique vertices: {stats['total_unique_vertices']}")
    logger.log(f"Total pixel observations: {stats['total_observations']}")
    logger.log(f"Average visibility: {stats['visibility']['mean']:.2f} frames per vertex")
    logger.log(f"Max visibility: {stats['visibility']['max']} frames")
    logger.log(f"Vertices visible in all frames: {stats['coverage']['visible_in_all_frames']}")
    logger.log(f"Vertices visible in 50%+ frames: {stats['coverage']['visible_in_50pct_frames']}")
    
    # Verify results
    logger.section("VERIFICATION")
    
    if stats['total_unique_vertices'] < 100:
        logger.warning(f"Low vertex count ({stats['total_unique_vertices']}). Check rendering!")
    else:
        logger.success(f"Found {stats['total_unique_vertices']} unique vertices")
    
    if stats['coverage']['visible_in_50pct_frames'] < 10:
        logger.warning("Few vertices visible in 50%+ frames. Surface may be highly occluded.")
    else:
        logger.success(f"{stats['coverage']['visible_in_50pct_frames']} vertices visible in 50%+ frames")
    
    # Save results
    logger.section("SAVING RESULTS")
    CORRESPONDENCE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save correspondence map
    corr_path = CORRESPONDENCE_DIR / "correspondence_map.json"
    with open(corr_path, 'w') as f:
        json.dump(corr_map, f)  # No indent for smaller file
    
    file_size_mb = corr_path.stat().st_size / (1024 * 1024)
    logger.log(f"Saved correspondence map: {corr_path} ({file_size_mb:.2f} MB)")
    
    # Save statistics
    stats_path = CORRESPONDENCE_DIR / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.log(f"Saved statistics: {stats_path}")
    
    # Summary
    logger.section("CORRESPONDENCE COMPLETE")
    logger.success(f"Unique vertices tracked: {stats['total_unique_vertices']}")
    logger.success(f"Total observations: {stats['total_observations']}")
    logger.success(f"Output: {CORRESPONDENCE_DIR}")
    
    # Print example correspondence
    logger.log("\nExample: First 3 vertices and their appearances:")
    count = 0
    for vertex_key, pixels in corr_map["vertex_to_pixels"].items():
        if count >= 3:
            break
        if len(pixels) >= 5:  # Only show well-visible vertices
            logger.log(f"  Vertex {vertex_key}: visible in {len(pixels)} frames")
            logger.log(f"    First 3 appearances: {pixels[:3]}")
            count += 1


if __name__ == "__main__":
    main()

