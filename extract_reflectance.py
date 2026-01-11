#!/usr/bin/env python3
"""
================================================================================
SCRIPT 3: EXTRACT_REFLECTANCE.PY - Reflectance Extraction and 2D Mapping
================================================================================

Purpose:
    Extract reflectance functions for tracked surface points and create
    2D reflectance maps for visualization and verification.

What is a Reflectance Function?
    For a surface point, the reflectance function describes how its
    brightness changes as the lighting conditions change. In our case,
    as the object rotates, the effective light direction relative to
    each surface point changes, creating a reflectance curve.

The Lambertian Model:
    For diffuse surfaces: I = max(0, n · l)
    - I = intensity (brightness) in [0, 1]
    - n = surface normal (unit vector)
    - l = light direction (unit vector)
    - max(0, ...) = no negative light (shadowed regions)

2D Reflectance Maps:
    Instead of 1D curves, we create 2D heatmaps where:
    - Rows = different surface points
    - Columns = rotation angles
    - Color = intensity value

    This provides a powerful visual comparison between measured and
    ground truth values.

Run:
    python extract_reflectance.py

Input:
    renders/theta_XXX.exr           - Lit images
    renders/theta_XXX_normal.exr    - Normal maps
    correspondence/correspondence_map.json

Output:
    output/reflectance_map_2d.png   - 2D reflectance heatmap
    output/verification.png         - Detailed verification plots
    output/results.json             - Numerical results
    logs/reflectance.log            - Execution log

Author: Abel & Team
Date: January 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Try to import EXR reading library
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
RENDERS_DIR = SCRIPT_DIR / "renders"
CORRESPONDENCE_DIR = SCRIPT_DIR / "correspondence"
OUTPUT_DIR = SCRIPT_DIR / "output"
DOCS_DIR = SCRIPT_DIR / "docs"
LOGS_DIR = SCRIPT_DIR / "logs"

# Number of surface points to analyze (paper-ready: 100-300)
NUM_TEST_POINTS = 100

# Minimum visibility threshold (fraction of total frames)
MIN_VISIBILITY_FRACTION = 0.2  # At least 20% of frames

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class Logger:
    """Logging system for tracking reflectance extraction progress."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"REFLECTANCE EXTRACTION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def section(self, title: str):
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)
    
    def success(self, message: str):
        self.log(message, "SUCCESS")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")

logger = Logger(LOGS_DIR / "reflectance.log")

# ============================================================================
# IMAGE READING
# ============================================================================

def read_exr(filepath: Path) -> np.ndarray:
    """
    Read an EXR image file.
    
    Args:
        filepath: Path to the EXR file
    
    Returns:
        numpy array of shape (height, width, 3)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"EXR file not found: {filepath}")
    
    img = imageio.imread(str(filepath))
    return img[:, :, :3] if img.shape[-1] > 3 else img


def decode_normal(rgb: np.ndarray) -> np.ndarray:
    """
    Decode RGB color to world-space normal vector.
    
    The Encoding (from render.py):
        RGB = (normal + 1) / 2
        
    The Decoding:
        normal = RGB * 2 - 1
    
    Args:
        rgb: Array of [R, G, B] values in [0, 1]
    
    Returns:
        Unit normal vector [nx, ny, nz]
    """
    normal = rgb * 2.0 - 1.0
    
    # Normalize to ensure unit length
    length = np.linalg.norm(normal)
    if length > 0.001:
        normal = normal / length
    
    return normal


# ============================================================================
# REFLECTANCE EXTRACTION
# ============================================================================

def load_light_direction() -> np.ndarray:
    """
    Load the light direction from render metadata.
    
    Why from Metadata:
        The light direction must match exactly what was used during
        rendering to compute accurate ground truth values.
    
    Returns:
        Normalized light direction vector
    """
    metadata_path = RENDERS_DIR / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    light_dir = np.array(metadata["light_direction"])
    return light_dir / np.linalg.norm(light_dir)


def extract_reflectance_for_vertex(
    vertex_key: str,
    pixel_list: List[List[int]],
    renders_dir: Path,
    light_direction: np.ndarray
) -> Dict[str, Any]:
    """
    Extract reflectance function for a single surface point.
    
    Algorithm:
        For each frame where the vertex is visible:
            1. Get the pixel location from correspondence
            2. Read measured intensity from lit image
            3. Read normal from normal map
            4. Compute ground truth: max(0, n · l)
    
    Args:
        vertex_key: String identifier for the vertex
        pixel_list: List of [frame, y, x] where vertex is visible
        renders_dir: Path to rendered images
        light_direction: Normalized light direction vector
    
    Returns:
        Dictionary with measured and ground truth reflectance data
    """
    measured = []
    ground_truth = []
    frames = []
    normals = []
    
    # Track which frames we've already processed
    seen_frames = set()
    
    for frame_idx, y, x in pixel_list:
        if frame_idx in seen_frames:
            continue
        seen_frames.add(frame_idx)
        
        try:
            # Load lit image and sample at corresponding pixel
            lit_path = renders_dir / f"theta_{frame_idx:03d}.exr"
            lit_img = read_exr(lit_path)
            intensity = lit_img[y, x, 0]  # Grayscale value
            
            # Load normal map and decode normal
            normal_path = renders_dir / f"theta_{frame_idx:03d}_normal.exr"
            normal_img = read_exr(normal_path)
            normal = decode_normal(normal_img[y, x, :3])
            
            # Compute ground truth: I = max(0, n · l)
            n_dot_l = np.dot(normal, light_direction)
            gt = max(0.0, n_dot_l)
            
            measured.append(intensity)
            ground_truth.append(gt)
            frames.append(frame_idx)
            normals.append(normal.tolist())
            
        except Exception as e:
            logger.warning(f"Error processing frame {frame_idx}: {e}")
    
    return {
        "vertex_key": vertex_key,
        "frames": frames,
        "measured": measured,
        "ground_truth": ground_truth,
        "normals": normals,
        "num_observations": len(measured)
    }


def compute_metrics(measured: List[float], ground_truth: List[float]) -> Dict[str, float]:
    """
    Compute verification metrics between measured and ground truth.
    
    Metrics:
        - MSE: Mean Squared Error - lower is better
        - MAE: Mean Absolute Error - lower is better
        - Correlation: Pearson correlation - higher is better (max 1.0)
    
    Why These Metrics:
        - MSE penalizes large errors heavily
        - MAE gives intuitive average error magnitude
        - Correlation shows if the pattern matches (even if scaled)
    
    Special Case:
        If both measured and ground truth are constant (std=0), but equal,
        we set correlation = 1.0 (perfect match). This happens for surface
        points that are always in shadow or always fully lit.
    
    Args:
        measured: List of measured intensity values
        ground_truth: List of ground truth values
    
    Returns:
        Dictionary of metrics
    """
    measured = np.array(measured)
    ground_truth = np.array(ground_truth)
    
    if len(measured) < 2:
        return {"mse": np.nan, "mae": np.nan, "correlation": np.nan, "valid": False}
    
    # MSE and MAE
    errors = measured - ground_truth
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    
    # Pearson correlation
    std_m = np.std(measured)
    std_g = np.std(ground_truth)
    
    if std_m > 1e-8 and std_g > 1e-8:
        # Normal case: both have variation
        correlation = np.corrcoef(measured, ground_truth)[0, 1]
        valid = True
    elif std_m < 1e-8 and std_g < 1e-8:
        # Both constant: if they match, correlation = 1.0
        if mse < 1e-6:
            correlation = 1.0  # Both constant and equal
            valid = True
        else:
            correlation = 0.0  # Both constant but different
            valid = False
    else:
        # One varies, one constant: undefined correlation
        correlation = np.nan
        valid = False
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "max_error": float(np.max(np.abs(errors))),
        "valid": valid
    }


# ============================================================================
# 2D REFLECTANCE MAP VISUALIZATION
# ============================================================================

def create_2d_reflectance_map(
    all_results: List[Dict[str, Any]],
    theta_samples: int,
    output_path: Path
):
    """
    Create 2D reflectance map visualization.
    
    The 2D Map:
        - Each row = one surface point
        - Each column = one rotation angle
        - Color intensity = reflectance value
        
        This shows at a glance:
        - How reflectance varies with rotation
        - Whether measured matches ground truth
        - Which surface points have similar behavior
    
    Why 2D Instead of 1D:
        - Easier to compare many points at once
        - Patterns become visually apparent
        - More paper-ready visualization
    
    Args:
        all_results: List of reflectance data for each test point
        theta_samples: Total number of rotation angles
        output_path: Where to save the figure
    """
    num_points = len(all_results)
    
    # Create arrays for 2D map
    # Shape: (num_points, theta_samples)
    measured_map = np.full((num_points, theta_samples), np.nan)
    gt_map = np.full((num_points, theta_samples), np.nan)
    
    # Fill in the data
    for i, result in enumerate(all_results):
        for j, (frame, meas, gt) in enumerate(zip(
            result["frames"], result["measured"], result["ground_truth"]
        )):
            measured_map[i, frame] = meas
            gt_map[i, frame] = gt
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    
    fig.suptitle(
        '2D Reflectance Maps: Measured vs Ground Truth\n'
        f'{num_points} Surface Points x {theta_samples} Rotation Angles',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Layout: 2x2 grid + statistics
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.6], hspace=0.3, wspace=0.2)
    
    # Panel 1: Measured reflectance
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(measured_map, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Measured Reflectance (Our System)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rotation Angle Index')
    ax1.set_ylabel('Surface Point Index')
    plt.colorbar(im1, ax=ax1, label='Intensity', shrink=0.8)
    
    # Panel 2: Ground truth
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(gt_map, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('Ground Truth (Lambertian: max(0, n·l))', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Rotation Angle Index')
    ax2.set_ylabel('Surface Point Index')
    plt.colorbar(im2, ax=ax2, label='Intensity', shrink=0.8)
    
    # Panel 3: Difference map
    ax3 = fig.add_subplot(gs[1, 0])
    diff_map = measured_map - gt_map
    max_diff = np.nanmax(np.abs(diff_map))
    max_diff = max(0.01, max_diff)  # Ensure non-zero range
    im3 = ax3.imshow(diff_map, aspect='auto', cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    ax3.set_title('Difference (Measured - Ground Truth)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Rotation Angle Index')
    ax3.set_ylabel('Surface Point Index')
    plt.colorbar(im3, ax=ax3, label='Error', shrink=0.8)
    
    # Panel 4: Sample curves overlay
    ax4 = fig.add_subplot(gs[1, 1])
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_points)))
    
    for i in range(min(10, num_points)):
        result = all_results[i]
        frames = result["frames"]
        measured = result["measured"]
        ground_truth = result["ground_truth"]
        
        ax4.plot(frames, measured, 'o-', color=colors[i], 
                 label=f'Point {i+1}', markersize=3, linewidth=1)
        ax4.plot(frames, ground_truth, 's--', color=colors[i], 
                 alpha=0.5, markersize=2, linewidth=1)
    
    ax4.set_xlabel('Rotation Angle Index')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Sample Curves (Solid=Measured, Dashed=GT)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8, ncol=2)
    ax4.set_xlim(-1, theta_samples)
    ax4.set_ylim(-0.05, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Compute aggregate statistics (only from valid points with variation)
    all_mse = []
    all_corr = []
    for result in all_results:
        if len(result["measured"]) >= 2:
            metrics = compute_metrics(result["measured"], result["ground_truth"])
            if not np.isnan(metrics["mse"]):
                all_mse.append(metrics["mse"])
            # Only include correlation from valid points (with variation)
            if metrics.get("valid", True) and not np.isnan(metrics["correlation"]):
                all_corr.append(metrics["correlation"])
    
    avg_mse = np.mean(all_mse) if all_mse else np.nan
    avg_corr = np.mean(all_corr) if all_corr else np.nan
    
    # Create summary text
    summary_text = (
        f"VERIFICATION RESULTS\n"
        f"{'='*50}\n"
        f"Number of surface points analyzed: {num_points}\n"
        f"Average MSE:         {avg_mse:.8f}\n"
        f"Average Correlation: {avg_corr:.6f}\n"
        f"{'='*50}\n"
    )
    
    if avg_corr > 0.99:
        summary_text += "STATUS: EXCELLENT - Near-perfect match with physics!"
    elif avg_corr > 0.95:
        summary_text += "STATUS: GOOD - Strong agreement with physics."
    elif avg_corr > 0.8:
        summary_text += "STATUS: MODERATE - Some discrepancies present."
    else:
        summary_text += "STATUS: NEEDS REVIEW - Check correspondence and rendering."
    
    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
             fontsize=12, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.log(f"Saved 2D reflectance map: {output_path}")
    
    return avg_mse, avg_corr


def create_verification_plot(
    all_results: List[Dict[str, Any]],
    output_path: Path
):
    """
    Create detailed verification plots for a subset of points.
    
    For each point, we show:
        1. Reflectance curve (measured vs ground truth)
        2. Correlation scatter plot
        3. Error histogram
    
    Args:
        all_results: List of reflectance data
        output_path: Where to save the figure
    """
    # Select top 6 points by number of observations
    sorted_results = sorted(all_results, key=lambda x: x["num_observations"], reverse=True)
    top_results = sorted_results[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Reflectance Function Verification', fontsize=14, fontweight='bold')
    
    for idx, (ax, result) in enumerate(zip(axes.flatten(), top_results)):
        frames = np.array(result["frames"])
        measured = np.array(result["measured"])
        ground_truth = np.array(result["ground_truth"])
        
        # Plot curves
        ax.plot(frames, measured, 'b-o', label='Measured', markersize=4, linewidth=1.5)
        ax.plot(frames, ground_truth, 'r--s', label='Ground Truth', markersize=3, linewidth=1.5)
        
        # Compute metrics
        metrics = compute_metrics(result["measured"], result["ground_truth"])
        
        ax.set_xlabel('Rotation Angle Index')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Point {idx+1}: Corr={metrics["correlation"]:.4f}, MSE={metrics["mse"]:.6f}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.log(f"Saved verification plot: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function to extract reflectance and create visualizations.
    
    Pipeline Steps:
        1. Load correspondence map
        2. Select test points (well-visible vertices)
        3. Extract reflectance for each point
        4. Compute verification metrics
        5. Create 2D reflectance maps
        6. Save results
    """
    logger.section("REFLECTANCE EXTRACTION AND VERIFICATION")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load correspondence map
    logger.log("Loading correspondence map...")
    corr_path = CORRESPONDENCE_DIR / "correspondence_map.json"
    
    if not corr_path.exists():
        logger.error(f"Correspondence map not found: {corr_path}")
        logger.error("Please run build_correspondence.py first!")
        return
    
    with open(corr_path, 'r') as f:
        corr_map = json.load(f)
    
    theta_samples = corr_map["metadata"]["theta_samples"]
    total_vertices = corr_map["metadata"]["total_vertices"]
    
    logger.log(f"Loaded {total_vertices} vertices, {theta_samples} frames")
    
    # Step 2: Load light direction
    light_direction = load_light_direction()
    logger.log(f"Light direction: ({light_direction[0]:.4f}, {light_direction[1]:.4f}, {light_direction[2]:.4f})")
    
    # Step 3: Select test points
    logger.section("SELECTING TEST POINTS")
    
    min_visibility = int(theta_samples * MIN_VISIBILITY_FRACTION)
    logger.log(f"Minimum visibility threshold: {min_visibility} frames ({MIN_VISIBILITY_FRACTION*100:.0f}%)")
    
    # Sort vertices by visibility count
    sorted_vertices = sorted(
        corr_map["vertex_to_pixels"].items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    # Filter by minimum visibility and take top N
    valid_vertices = [(k, v) for k, v in sorted_vertices if len(v) >= min_visibility]
    test_vertices = valid_vertices[:NUM_TEST_POINTS]
    
    logger.log(f"Vertices meeting threshold: {len(valid_vertices)}")
    logger.log(f"Selected for analysis: {len(test_vertices)}")
    
    if len(test_vertices) < 5:
        logger.warning("Few valid test points! Results may not be representative.")
    
    # Step 4: Extract reflectance functions
    logger.section("EXTRACTING REFLECTANCE FUNCTIONS")
    
    all_results = []
    start_time = datetime.now()
    
    for i, (vertex_key, pixel_list) in enumerate(test_vertices):
        if (i + 1) % 20 == 0 or i == 0:
            logger.log(f"Processing point {i+1}/{len(test_vertices)}...")
        
        result = extract_reflectance_for_vertex(
            vertex_key, pixel_list, RENDERS_DIR, light_direction
        )
        
        if result["num_observations"] >= 2:
            metrics = compute_metrics(result["measured"], result["ground_truth"])
            result["metrics"] = metrics
            all_results.append(result)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.log(f"Extraction completed in {elapsed:.1f} seconds")
    logger.log(f"Valid results: {len(all_results)}/{len(test_vertices)}")
    
    # Step 5: Create visualizations
    logger.section("CREATING VISUALIZATIONS")
    
    # 2D reflectance map
    avg_mse, avg_corr = create_2d_reflectance_map(
        all_results, theta_samples, OUTPUT_DIR / "reflectance_map_2d.png"
    )
    
    # Also save to docs folder
    create_2d_reflectance_map(
        all_results, theta_samples, DOCS_DIR / "reflectance_map_2d.png"
    )
    
    # Verification plot
    create_verification_plot(all_results, OUTPUT_DIR / "verification.png")
    create_verification_plot(all_results, DOCS_DIR / "verification.png")
    
    # Step 6: Save numerical results
    logger.section("SAVING RESULTS")
    
    # Aggregate statistics (only from valid points)
    all_metrics = [r["metrics"] for r in all_results if "metrics" in r]
    valid_metrics = [m for m in all_metrics if m.get("valid", True)]
    
    logger.log(f"Valid points for correlation: {len(valid_metrics)}/{len(all_metrics)}")
    
    results_summary = {
        "num_points_analyzed": len(all_results),
        "num_valid_points": len(valid_metrics),
        "theta_samples": theta_samples,
        "aggregate": {
            "avg_mse": float(np.mean([m["mse"] for m in valid_metrics])) if valid_metrics else 0.0,
            "avg_mae": float(np.mean([m["mae"] for m in valid_metrics])) if valid_metrics else 0.0,
            "avg_correlation": float(np.mean([m["correlation"] for m in valid_metrics])) if valid_metrics else 0.0,
            "min_correlation": float(np.min([m["correlation"] for m in valid_metrics])) if valid_metrics else 0.0,
            "max_correlation": float(np.max([m["correlation"] for m in valid_metrics])) if valid_metrics else 0.0
        },
        "per_point": [
            {
                "vertex_key": r["vertex_key"],
                "num_observations": r["num_observations"],
                "mse": r["metrics"]["mse"],
                "correlation": r["metrics"]["correlation"]
            }
            for r in all_results if "metrics" in r
        ]
    }
    
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.log(f"Saved results: {results_path}")
    
    # Summary
    logger.section("VERIFICATION COMPLETE")
    logger.success(f"Points analyzed: {len(all_results)}")
    logger.success(f"Average MSE: {avg_mse:.8f}")
    logger.success(f"Average Correlation: {avg_corr:.6f}")
    
    if avg_corr > 0.99:
        logger.success("VERIFICATION PASSED - Near-perfect match with physics!")
    elif avg_corr > 0.95:
        logger.success("VERIFICATION PASSED - Strong agreement with physics.")
    elif avg_corr > 0.8:
        logger.warning("VERIFICATION MODERATE - Some discrepancies present.")
    else:
        logger.error("VERIFICATION NEEDS REVIEW - Check pipeline components.")
    
    logger.log(f"\nOutput files:")
    logger.log(f"  - {OUTPUT_DIR / 'reflectance_map_2d.png'}")
    logger.log(f"  - {OUTPUT_DIR / 'verification.png'}")
    logger.log(f"  - {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()

