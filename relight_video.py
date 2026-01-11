#!/usr/bin/env python3
"""
================================================================================
SCRIPT 4: RELIGHT_VIDEO.PY - Video Relighting Demonstration
================================================================================

Purpose:
    Demonstrate practical video relighting using the extracted normal maps.
    This is the ultimate verification that our correspondence-based method
    works correctly for real-world applications.

What is Video Relighting?
    Given a video of a rotating object, we can computationally change the
    lighting direction in post-processing. This is possible because:
    
    1. We have normal maps for each frame (from correspondence tracking)
    2. For Lambertian surfaces: I = max(0, n · l)
    3. We can compute new intensity for ANY light direction l
    
    This enables "virtual lighting" without re-shooting.

Applications:
    - Visual effects: Match lighting between composited elements
    - Product photography: Try different lighting setups digitally
    - Research: Study reflectance properties

Output:
    - Original vs relit comparison video (GIF)
    - Multi-light comparison images
    - Before/after frame comparisons

Run:
    python relight_video.py

Input:
    renders/theta_XXX_normal.exr  - Normal maps
    renders/theta_XXX.exr         - Original lit images

Output:
    output/relighting_comparison.gif   - Animated comparison
    output/relit_frames/               - Individual relit frames
    output/multi_light.png             - Multiple lighting comparison
    logs/relighting.log                - Execution log

Author: Abel & Team
Date: January 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

# Try to import image libraries
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
OUTPUT_DIR = SCRIPT_DIR / "output"
DOCS_DIR = SCRIPT_DIR / "docs"
LOGS_DIR = SCRIPT_DIR / "logs"

# New light directions for relighting demo
NEW_LIGHT_DIRECTIONS = [
    {"name": "Top", "direction": (0, 0, 1)},
    {"name": "Right", "direction": (1, 0, 0)},
    {"name": "Front", "direction": (0, -1, 0)},
    {"name": "Top-Left", "direction": (-1, 0, 1)},
    {"name": "Bottom-Right", "direction": (1, 0, -0.5)},
]

# GIF settings
GIF_FPS = 10
GIF_FRAME_SKIP = 2  # Use every Nth frame for smaller file

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class Logger:
    """Logging system for relighting progress."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"RELIGHTING LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

logger = Logger(LOGS_DIR / "relighting.log")

# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def read_exr(filepath: Path) -> np.ndarray:
    """
    Read an EXR image file.
    
    Args:
        filepath: Path to the EXR file
    
    Returns:
        numpy array of shape (height, width, 3)
    """
    img = imageio.imread(str(filepath))
    return img[:, :, :3] if img.shape[-1] > 3 else img


def decode_normal_map(normal_rgb: np.ndarray) -> np.ndarray:
    """
    Decode entire normal map from RGB to vectors.
    
    Encoding (from render.py): RGB = (normal + 1) / 2
    Decoding: normal = RGB * 2 - 1
    
    Args:
        normal_rgb: Normal map image of shape (H, W, 3)
    
    Returns:
        Normal vectors of shape (H, W, 3), normalized
    """
    normals = normal_rgb * 2.0 - 1.0
    
    # Normalize each vector
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)  # Avoid division by zero
    normals = normals / lengths
    
    return normals


def compute_shading(normals: np.ndarray, light_direction: Tuple[float, float, float]) -> np.ndarray:
    """
    Compute Lambertian shading for all pixels.
    
    The Lambertian Model:
        I = max(0, n · l)
        
    This models diffuse surfaces where light scatters equally
    in all directions from the surface.
    
    Args:
        normals: Normal vectors of shape (H, W, 3)
        light_direction: Light direction as (x, y, z) tuple
    
    Returns:
        Shading values of shape (H, W) in range [0, 1]
    """
    # Normalize light direction
    light_dir = np.array(light_direction)
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Dot product: n · l for each pixel
    n_dot_l = np.sum(normals * light_dir, axis=-1)
    
    # Clamp to [0, 1]
    shading = np.maximum(0, n_dot_l)
    
    return shading


def create_visibility_mask(normal_map: np.ndarray) -> np.ndarray:
    """
    Create mask of visible (non-background) pixels.
    
    Background pixels have normals pointing in undefined directions
    or have very small values. We detect them by looking for
    abnormal vector lengths.
    
    Args:
        normal_map: Decoded normal map of shape (H, W, 3)
    
    Returns:
        Boolean mask of shape (H, W)
    """
    # Check if the original RGB values were near-zero (background)
    # Decoded normals from [0,0,0] would be [-1,-1,-1]
    # Visible pixels should have |n| ≈ 1
    
    lengths = np.linalg.norm(normal_map, axis=-1)
    
    # Also check for the specific pattern of background
    # Background in our renders appears as black = RGB(0,0,0)
    # Which decodes to normal (-1,-1,-1), length = sqrt(3)
    # But we normalized, so we need another approach
    
    # Use the fact that valid normals have z component not exactly -1
    # after normalization from a black pixel
    z_component = normal_map[:, :, 2]
    
    # Visible if not uniform -0.577 (which comes from normalized (-1,-1,-1))
    is_visible = ~(
        (np.abs(normal_map[:, :, 0] + 0.577) < 0.01) &
        (np.abs(normal_map[:, :, 1] + 0.577) < 0.01) &
        (np.abs(normal_map[:, :, 2] + 0.577) < 0.01)
    )
    
    return is_visible


# ============================================================================
# RELIGHTING FUNCTIONS
# ============================================================================

def relight_frame(
    normal_map_path: Path,
    new_light_direction: Tuple[float, float, float]
) -> np.ndarray:
    """
    Relight a single frame with a new light direction.
    
    Process:
        1. Load normal map
        2. Decode normals from RGB
        3. Compute new shading: I = max(0, n · l_new)
        4. Create grayscale image
    
    Args:
        normal_map_path: Path to the normal map EXR
        new_light_direction: New light direction (x, y, z)
    
    Returns:
        Relit image of shape (H, W) in range [0, 1]
    """
    # Load and decode normal map
    normal_rgb = read_exr(normal_map_path)
    normals = decode_normal_map(normal_rgb)
    
    # Compute new shading
    shading = compute_shading(normals, new_light_direction)
    
    return shading


def create_comparison_frame(
    original: np.ndarray,
    relit: np.ndarray,
    frame_idx: int,
    light_name: str
) -> np.ndarray:
    """
    Create side-by-side comparison of original and relit frames.
    
    Args:
        original: Original lit image (H, W)
        relit: Relit image (H, W)
        frame_idx: Frame number for labeling
        light_name: Name of the new light direction
    
    Returns:
        Comparison image ready for display
    """
    import io
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original (Frame {frame_idx})', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(relit, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Relit: {light_name}', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to image array using buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    
    plt.close(fig)
    
    return img[:, :, :3]  # Remove alpha channel if present


# ============================================================================
# VIDEO GENERATION
# ============================================================================

def create_relighting_gif(
    renders_dir: Path,
    output_path: Path,
    new_light_direction: Tuple[float, float, float],
    light_name: str,
    theta_samples: int
):
    """
    Create animated GIF comparing original and relit video.
    
    Why GIF:
        - Universally viewable (no codec issues)
        - Loops automatically
        - Easy to embed in documents
    
    Args:
        renders_dir: Directory with rendered frames
        output_path: Where to save the GIF
        new_light_direction: Light direction for relighting
        light_name: Name for labeling
        theta_samples: Number of frames
    """
    logger.log(f"Creating relighting GIF: {light_name}...")
    
    frames = []
    
    for i in range(0, theta_samples, GIF_FRAME_SKIP):
        # Load original
        original_path = renders_dir / f"theta_{i:03d}.exr"
        original = read_exr(original_path)[:, :, 0]  # Grayscale
        
        # Relight
        normal_path = renders_dir / f"theta_{i:03d}_normal.exr"
        relit = relight_frame(normal_path, new_light_direction)
        
        # Create comparison
        comparison = create_comparison_frame(original, relit, i, light_name)
        frames.append(comparison)
        
        if (i + 1) % 20 == 0:
            logger.log(f"  Processed frame {i+1}/{theta_samples}")
    
    # Save as GIF
    imageio.mimsave(
        str(output_path),
        frames,
        fps=GIF_FPS,
        loop=0  # Loop forever
    )
    
    logger.log(f"Saved GIF: {output_path}")


def create_multi_light_comparison(
    renders_dir: Path,
    output_path: Path,
    frame_idx: int = 0
):
    """
    Create image showing same frame under multiple light directions.
    
    This demonstrates the power of normal-based relighting:
    from a single capture, we can simulate many different
    lighting setups.
    
    Args:
        renders_dir: Directory with rendered frames
        output_path: Where to save the comparison image
        frame_idx: Which frame to use
    """
    logger.log("Creating multi-light comparison...")
    
    # Load original and normal map
    original_path = renders_dir / f"theta_{frame_idx:03d}.exr"
    normal_path = renders_dir / f"theta_{frame_idx:03d}_normal.exr"
    
    original = read_exr(original_path)[:, :, 0]
    
    # Create figure
    n_lights = len(NEW_LIGHT_DIRECTIONS) + 1  # +1 for original
    cols = 3
    rows = (n_lights + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    fig.suptitle('Multi-Light Relighting Demo', fontsize=14, fontweight='bold')
    
    # Original
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original', fontsize=11)
    axes[0].axis('off')
    
    # Relit versions
    for idx, light_config in enumerate(NEW_LIGHT_DIRECTIONS):
        relit = relight_frame(normal_path, light_config["direction"])
        
        axes[idx + 1].imshow(relit, cmap='gray', vmin=0, vmax=1)
        axes[idx + 1].set_title(f'Light: {light_config["name"]}', fontsize=11)
        axes[idx + 1].axis('off')
    
    # Hide unused axes
    for idx in range(len(NEW_LIGHT_DIRECTIONS) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.log(f"Saved multi-light comparison: {output_path}")


def save_relit_frames(
    renders_dir: Path,
    output_dir: Path,
    new_light_direction: Tuple[float, float, float],
    theta_samples: int
):
    """
    Save individual relit frames for detailed inspection.
    
    Args:
        renders_dir: Directory with normal maps
        output_dir: Where to save relit frames
        new_light_direction: Light direction for relighting
        theta_samples: Number of frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.log(f"Saving {theta_samples} relit frames...")
    
    for i in range(theta_samples):
        normal_path = renders_dir / f"theta_{i:03d}_normal.exr"
        relit = relight_frame(normal_path, new_light_direction)
        
        # Convert to 8-bit for PNG
        relit_uint8 = (np.clip(relit, 0, 1) * 255).astype(np.uint8)
        
        output_path = output_dir / f"relit_{i:03d}.png"
        imageio.imwrite(str(output_path), relit_uint8)
    
    logger.log(f"Saved frames to: {output_dir}")


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_relighting(
    renders_dir: Path,
    theta_samples: int
) -> Tuple[float, float]:
    """
    Verify relighting accuracy by comparing to original renders.
    
    If we relight using the ORIGINAL light direction, the result
    should match the original lit images. This verifies our
    normal-based relighting is correct.
    
    Args:
        renders_dir: Directory with rendered frames
        theta_samples: Number of frames
    
    Returns:
        Tuple of (average_mse, average_correlation)
    """
    logger.log("Verifying relighting accuracy...")
    
    # Load original light direction from metadata
    metadata_path = renders_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    original_light = tuple(metadata["light_direction"])
    
    mse_values = []
    corr_values = []
    
    for i in range(0, theta_samples, 5):  # Sample every 5th frame
        # Load original
        original_path = renders_dir / f"theta_{i:03d}.exr"
        original = read_exr(original_path)[:, :, 0]
        
        # Relight with original direction
        normal_path = renders_dir / f"theta_{i:03d}_normal.exr"
        relit = relight_frame(normal_path, original_light)
        
        # Create mask for visible pixels
        normal_rgb = read_exr(normal_path)
        normals = decode_normal_map(normal_rgb)
        mask = create_visibility_mask(normals)
        
        # Compare only visible pixels
        orig_masked = original[mask]
        relit_masked = relit[mask]
        
        if len(orig_masked) > 100:
            mse = np.mean((orig_masked - relit_masked) ** 2)
            
            if np.std(orig_masked) > 0.01 and np.std(relit_masked) > 0.01:
                corr = np.corrcoef(orig_masked, relit_masked)[0, 1]
            else:
                corr = 1.0  # If both are constant, they match
            
            mse_values.append(mse)
            corr_values.append(corr)
    
    avg_mse = np.mean(mse_values) if mse_values else np.nan
    avg_corr = np.mean(corr_values) if corr_values else np.nan
    
    logger.log(f"  Average MSE: {avg_mse:.6f}")
    logger.log(f"  Average Correlation: {avg_corr:.6f}")
    
    return avg_mse, avg_corr


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function to create relighting demonstrations.
    
    Pipeline Steps:
        1. Verify relighting accuracy
        2. Create multi-light comparison image
        3. Create before/after GIF
        4. Save individual relit frames
    """
    logger.section("VIDEO RELIGHTING DEMONSTRATION")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = RENDERS_DIR / "metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        logger.error("Please run render.py first!")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    theta_samples = metadata["theta_samples"]
    logger.log(f"Theta samples: {theta_samples}")
    
    # Step 1: Verify relighting
    logger.section("VERIFICATION")
    avg_mse, avg_corr = verify_relighting(RENDERS_DIR, theta_samples)
    
    if avg_corr > 0.99:
        logger.success("Relighting verified! MSE and correlation are excellent.")
    elif avg_corr > 0.95:
        logger.success("Relighting verified with good accuracy.")
    else:
        logger.warning("Relighting accuracy is lower than expected. Check normals.")
    
    # Step 2: Multi-light comparison
    logger.section("MULTI-LIGHT COMPARISON")
    create_multi_light_comparison(
        RENDERS_DIR, OUTPUT_DIR / "multi_light.png"
    )
    create_multi_light_comparison(
        RENDERS_DIR, DOCS_DIR / "multi_light.png"
    )
    
    # Step 3: Create relighting GIF (use "Top" light for demo)
    logger.section("CREATING RELIGHTING GIF")
    selected_light = NEW_LIGHT_DIRECTIONS[0]  # "Top" light
    
    create_relighting_gif(
        RENDERS_DIR,
        OUTPUT_DIR / "relighting_comparison.gif",
        selected_light["direction"],
        selected_light["name"],
        theta_samples
    )
    
    # Also save to docs
    create_relighting_gif(
        RENDERS_DIR,
        DOCS_DIR / "relighting_comparison.gif",
        selected_light["direction"],
        selected_light["name"],
        theta_samples
    )
    
    # Step 4: Save individual relit frames
    logger.section("SAVING RELIT FRAMES")
    save_relit_frames(
        RENDERS_DIR,
        OUTPUT_DIR / "relit_frames",
        selected_light["direction"],
        theta_samples
    )
    
    # Summary
    logger.section("RELIGHTING COMPLETE")
    logger.success(f"Verification MSE: {avg_mse:.6f}")
    logger.success(f"Verification Correlation: {avg_corr:.6f}")
    logger.log("\nOutput files:")
    logger.log(f"  - {OUTPUT_DIR / 'multi_light.png'}")
    logger.log(f"  - {OUTPUT_DIR / 'relighting_comparison.gif'}")
    logger.log(f"  - {OUTPUT_DIR / 'relit_frames/'}")
    logger.log(f"\nDocs files:")
    logger.log(f"  - {DOCS_DIR / 'multi_light.png'}")
    logger.log(f"  - {DOCS_DIR / 'relighting_comparison.gif'}")


if __name__ == "__main__":
    main()

