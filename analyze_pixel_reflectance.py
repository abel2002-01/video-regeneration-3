#!/usr/bin/env python3
"""
================================================================================
PIXEL REFLECTANCE ANALYSIS
================================================================================

Purpose:
    Analyze reflectance for individual pixels across all rotation angles.
    Select pixels that show good light variation (not always in shadow).
    Create detailed comparison between measured and ground truth.

This script:
    1. Finds pixels with varied lighting (not always dark)
    2. Tracks their reflectance across all frames
    3. Creates detailed comparison plots
    4. Shows that our system matches physics perfectly

Run:
    python analyze_pixel_reflectance.py

Author: Abel & Team
Date: January 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
RENDERS_DIR = SCRIPT_DIR / "renders"
OUTPUT_DIR = SCRIPT_DIR / "output"
DOCS_DIR = SCRIPT_DIR / "docs"

# Number of pixels to analyze
NUM_PIXELS = 20

# Minimum brightness threshold (pixels must have some bright frames)
MIN_BRIGHTNESS = 0.3
MIN_VARIATION = 0.2  # Must have intensity range > 0.2

# ============================================================================
# UTILITIES
# ============================================================================

def read_exr(filepath: Path) -> np.ndarray:
    """Read EXR file."""
    img = imageio.imread(str(filepath))
    return img[:, :, :3] if img.shape[-1] > 3 else img


def decode_normal(rgb: np.ndarray) -> np.ndarray:
    """Decode normal from RGB."""
    normal = rgb * 2.0 - 1.0
    length = np.linalg.norm(normal)
    if length > 0.001:
        normal = normal / length
    return normal


def get_visibility_mask(lit_img: np.ndarray) -> np.ndarray:
    """Get mask of visible (non-background) pixels."""
    # Background is typically very dark
    return lit_img[:, :, 0] > 0.001


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("="*70)
    print("PIXEL REFLECTANCE ANALYSIS")
    print("="*70)
    
    # Load metadata
    metadata_path = RENDERS_DIR / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    theta_samples = metadata["theta_samples"]
    light_dir = np.array(metadata["light_direction"])
    resolution = metadata["resolution"]
    
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Theta samples: {theta_samples}")
    print(f"Light direction: {light_dir}")
    
    # Step 1: Find pixels with good light variation
    print("\n[1] Finding well-lit pixels with good variation...")
    
    # Load all lit images to find good pixels
    all_lit = []
    for i in range(theta_samples):
        lit_path = RENDERS_DIR / f"theta_{i:03d}.exr"
        lit_img = read_exr(lit_path)[:, :, 0]
        all_lit.append(lit_img)
    
    all_lit = np.stack(all_lit, axis=0)  # (frames, H, W)
    
    # Compute statistics per pixel
    max_intensity = np.max(all_lit, axis=0)  # Max intensity across frames
    min_intensity = np.min(all_lit, axis=0)  # Min intensity
    intensity_range = max_intensity - min_intensity  # Variation
    mean_intensity = np.mean(all_lit, axis=0)
    
    # Find pixels that:
    # 1. Have max intensity above threshold (get some light)
    # 2. Have good variation (intensity changes with rotation)
    good_pixels = (max_intensity > MIN_BRIGHTNESS) & (intensity_range > MIN_VARIATION)
    
    good_coords = np.argwhere(good_pixels)
    print(f"Found {len(good_coords)} pixels with good variation")
    
    if len(good_coords) < NUM_PIXELS:
        print("Not enough good pixels, lowering thresholds...")
        good_pixels = (max_intensity > 0.1) & (intensity_range > 0.1)
        good_coords = np.argwhere(good_pixels)
        print(f"Found {len(good_coords)} pixels with relaxed thresholds")
    
    # Select top N by variation
    variations = intensity_range[good_coords[:, 0], good_coords[:, 1]]
    top_indices = np.argsort(variations)[-NUM_PIXELS:]
    selected_pixels = good_coords[top_indices]
    
    print(f"Selected {len(selected_pixels)} pixels for analysis")
    
    # Step 2: Extract reflectance for each pixel
    print("\n[2] Extracting reflectance curves...")
    
    pixel_data = []
    
    for idx, (y, x) in enumerate(selected_pixels):
        measured = []
        ground_truth = []
        frames = []
        
        for frame_idx in range(theta_samples):
            # Load lit and normal
            lit_path = RENDERS_DIR / f"theta_{frame_idx:03d}.exr"
            normal_path = RENDERS_DIR / f"theta_{frame_idx:03d}_normal.exr"
            
            lit_img = read_exr(lit_path)
            normal_img = read_exr(normal_path)
            
            # Get values at this pixel
            intensity = lit_img[y, x, 0]
            normal = decode_normal(normal_img[y, x, :3])
            
            # Compute ground truth
            n_dot_l = np.dot(normal, light_dir)
            gt = max(0.0, n_dot_l)
            
            measured.append(intensity)
            ground_truth.append(gt)
            frames.append(frame_idx)
        
        # Compute metrics
        measured = np.array(measured)
        ground_truth = np.array(ground_truth)
        
        mse = np.mean((measured - ground_truth) ** 2)
        if np.std(measured) > 1e-8 and np.std(ground_truth) > 1e-8:
            corr = np.corrcoef(measured, ground_truth)[0, 1]
        else:
            corr = 1.0 if mse < 1e-6 else 0.0
        
        pixel_data.append({
            'y': int(y),
            'x': int(x),
            'frames': frames,
            'measured': measured,
            'ground_truth': ground_truth,
            'mse': mse,
            'correlation': corr,
            'max_intensity': float(np.max(measured)),
            'intensity_range': float(np.max(measured) - np.min(measured))
        })
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(selected_pixels)} pixels")
    
    # Step 3: Create visualization
    print("\n[3] Creating visualizations...")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    fig.suptitle(
        f'Pixel Reflectance Analysis: Measured vs Ground Truth\n'
        f'{len(selected_pixels)} Well-Lit Pixels × {theta_samples} Rotation Angles',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 0.8], hspace=0.3, wspace=0.2)
    
    # Panel 1: 2D Reflectance Map (Measured)
    ax1 = fig.add_subplot(gs[0, 0])
    measured_map = np.array([p['measured'] for p in pixel_data])
    im1 = ax1.imshow(measured_map, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_title('Measured Reflectance (Well-Lit Pixels)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Rotation Angle Index')
    ax1.set_ylabel('Pixel Index')
    plt.colorbar(im1, ax=ax1, label='Intensity', shrink=0.8)
    
    # Panel 2: 2D Ground Truth Map
    ax2 = fig.add_subplot(gs[0, 1])
    gt_map = np.array([p['ground_truth'] for p in pixel_data])
    im2 = ax2.imshow(gt_map, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('Ground Truth (Lambertian: max(0, n·l))', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Rotation Angle Index')
    ax2.set_ylabel('Pixel Index')
    plt.colorbar(im2, ax=ax2, label='Intensity', shrink=0.8)
    
    # Panel 3: Sample curves (top 6 pixels)
    ax3 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, min(6, len(pixel_data))))
    
    for i, (pdata, color) in enumerate(zip(pixel_data[:6], colors)):
        ax3.plot(pdata['frames'], pdata['measured'], '-o', color=color,
                 label=f'Pixel ({pdata["y"]},{pdata["x"]})', markersize=2, linewidth=1.5)
        ax3.plot(pdata['frames'], pdata['ground_truth'], '--', color=color,
                 alpha=0.6, linewidth=1.5)
    
    ax3.set_xlabel('Rotation Angle Index')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Sample Reflectance Curves (Solid=Measured, Dashed=GT)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim(-1, theta_samples)
    ax3.set_ylim(-0.05, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Correlation scatter
    ax4 = fig.add_subplot(gs[1, 1])
    all_measured = np.concatenate([p['measured'] for p in pixel_data])
    all_gt = np.concatenate([p['ground_truth'] for p in pixel_data])
    
    ax4.scatter(all_gt, all_measured, alpha=0.3, s=5, c='blue')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Match')
    ax4.set_xlabel('Ground Truth')
    ax4.set_ylabel('Measured')
    ax4.set_title('Correlation: All Pixels', fontsize=12, fontweight='bold')
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Compute overall correlation
    if np.std(all_measured) > 1e-8 and np.std(all_gt) > 1e-8:
        overall_corr = np.corrcoef(all_measured, all_gt)[0, 1]
    else:
        overall_corr = 1.0
    overall_mse = np.mean((all_measured - all_gt) ** 2)
    
    ax4.text(0.05, 0.95, f'Correlation: {overall_corr:.6f}\nMSE: {overall_mse:.8f}',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 5: Statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    avg_corr = np.mean([p['correlation'] for p in pixel_data])
    avg_mse = np.mean([p['mse'] for p in pixel_data])
    avg_max = np.mean([p['max_intensity'] for p in pixel_data])
    avg_range = np.mean([p['intensity_range'] for p in pixel_data])
    
    summary = (
        f"PIXEL REFLECTANCE ANALYSIS RESULTS\n"
        f"{'='*60}\n"
        f"Pixels analyzed: {len(pixel_data)}\n"
        f"Frames per pixel: {theta_samples}\n"
        f"Average max intensity: {avg_max:.3f}\n"
        f"Average intensity range: {avg_range:.3f}\n"
        f"{'='*60}\n"
        f"Average MSE: {avg_mse:.10f}\n"
        f"Average Correlation: {avg_corr:.6f}\n"
        f"Overall Correlation: {overall_corr:.6f}\n"
        f"{'='*60}\n"
    )
    
    if avg_corr > 0.99:
        summary += "STATUS: EXCELLENT - Perfect match with physics!"
    elif avg_corr > 0.95:
        summary += "STATUS: VERY GOOD - Strong agreement with physics."
    else:
        summary += "STATUS: Check results."
    
    ax5.text(0.5, 0.5, summary, transform=ax5.transAxes,
             fontsize=11, fontfamily='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = DOCS_DIR / "pixel_reflectance_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Also save to output
    plt.savefig(OUTPUT_DIR / "pixel_reflectance_analysis.png", dpi=150, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    # Create individual pixel detail plots
    print("\n[4] Creating detailed pixel plots...")
    
    fig2, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig2.suptitle('Individual Pixel Reflectance Curves', fontsize=14, fontweight='bold')
    
    for idx, (ax, pdata) in enumerate(zip(axes.flatten(), pixel_data)):
        ax.plot(pdata['frames'], pdata['measured'], 'b-o', markersize=2, linewidth=1, label='Measured')
        ax.plot(pdata['frames'], pdata['ground_truth'], 'r--', linewidth=1.5, label='Ground Truth')
        ax.set_title(f'Pixel ({pdata["y"]},{pdata["x"]})\nCorr={pdata["correlation"]:.4f}', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "pixel_reflectance_details.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / "pixel_reflectance_details.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: pixel_reflectance_details.png")
    
    plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Pixels analyzed: {len(pixel_data)}")
    print(f"Average correlation: {avg_corr:.6f}")
    print(f"Overall correlation: {overall_corr:.6f}")
    print(f"Average MSE: {avg_mse:.10f}")
    print(f"\nOutput files:")
    print(f"  - {DOCS_DIR / 'pixel_reflectance_analysis.png'}")
    print(f"  - {DOCS_DIR / 'pixel_reflectance_details.png'}")


if __name__ == "__main__":
    main()

