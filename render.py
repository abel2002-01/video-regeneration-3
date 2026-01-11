#!/usr/bin/env python3
"""
================================================================================
SCRIPT 1: RENDER.PY - Blender Rendering Pipeline
================================================================================

Purpose:
    Render a 3D object (Suzanne) rotating under fixed camera and lighting.
    For each rotation angle, we capture three types of images:
    1. Lit Image: Shows brightness under Lambertian shading (I = max(0, n·l))
    2. Vertex ID Map: Encodes unique vertex IDs as RGB colors for tracking
    3. Normal Map: Stores world-space surface normals as RGB

Mathematical Foundation:
    - Lambertian reflectance: I = max(0, n · l)
    - Where n = surface normal (unit vector), l = light direction (unit vector)
    - This models how diffuse surfaces reflect light

Why This Works:
    - By rotating the object, each surface point changes its world-space normal
    - The vertex ID stays constant (rotation-invariant identifier)
    - We can track the same physical surface point across all frames

Run:
    blender --background --python render.py

Output:
    renders/theta_XXX.exr           - Lit images (36 frames)
    renders/theta_XXX_vertexid.exr  - Vertex ID maps (36 frames)
    renders/theta_XXX_normal.exr    - Normal maps (36 frames)
    renders/metadata.json           - Rendering parameters
    logs/render.log                 - Detailed execution log

Author: Abel & Team
Date: January 2026
================================================================================
"""

import bpy
import bmesh
import math
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

# ============================================================================
# CONFIGURATION - Paper-Ready Parameters
# ============================================================================

# Resolution: Higher = more detail, but slower
RESOLUTION = 512  # 512x512 for paper quality (increase to 1024 for final)

# Number of rotation samples: More = smoother reflectance curves
THETA_SAMPLES = 72  # 72 samples = 5° increments (increase to 360 for 1° steps)

# Subdivision level: Higher = more vertices = better correspondence
SUBDIVISION_LEVEL = 3  # Level 3 gives ~32,000 vertices

# Camera distance: Affects object size in frame
CAMERA_DISTANCE = 5.0

# Light direction (normalized): Defines the illumination angle
LIGHT_DIRECTION = (1.0, 1.0, 1.0)  # Diagonal from top-right-front

# Output directories
SCRIPT_DIR = Path(__file__).parent
RENDERS_DIR = SCRIPT_DIR / "renders"
LOGS_DIR = SCRIPT_DIR / "logs"

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class Logger:
    """
    Comprehensive logging system that writes to both console and file.
    
    Why logging matters:
        - Track execution progress for long renders
        - Debug issues by reviewing the log
        - Document the exact parameters used
        - Verify each step completed correctly
    """
    
    def __init__(self, log_file: Path):
        """
        Initialize logger with output file.
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear previous log
        with open(self.log_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"RENDER LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message to both console and file.
        
        Args:
            message: The message to log
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        # Console output
        print(formatted)
        
        # File output
        with open(self.log_file, 'a') as f:
            f.write(formatted + "\n")
    
    def section(self, title: str):
        """Log a section header for visual organization."""
        separator = "=" * 70
        self.log(separator)
        self.log(title)
        self.log(separator)
    
    def step(self, step_num: int, total: int, description: str):
        """Log a pipeline step."""
        self.log(f"[Step {step_num}/{total}] {description}")
    
    def success(self, message: str):
        """Log a success message."""
        self.log(message, "SUCCESS")
    
    def error(self, message: str):
        """Log an error message."""
        self.log(message, "ERROR")

# Initialize logger
logger = Logger(LOGS_DIR / "render.log")

# ============================================================================
# SCENE SETUP FUNCTIONS
# ============================================================================

def clear_scene():
    """
    Remove all objects from the Blender scene.
    
    Why:
        We start with a clean slate to ensure reproducibility.
        Any leftover objects from previous runs could interfere.
    """
    logger.log("Clearing scene of all existing objects...")
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Also clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    
    logger.log("Scene cleared successfully")


def create_suzanne() -> bpy.types.Object:
    """
    Create the Suzanne (monkey head) mesh with subdivision.
    
    Why Suzanne:
        - Complex geometry with varied surface normals
        - Standard Blender test mesh, reproducible
        - Has front/back/sides for good occlusion testing
    
    Why Subdivision:
        - More vertices = more precise correspondence tracking
        - Reduces interpolation artifacts in vertex ID maps
        - Each vertex covers approximately 1 pixel
    
    Returns:
        The created Suzanne object
    """
    logger.log(f"Creating Suzanne mesh with subdivision level {SUBDIVISION_LEVEL}...")
    
    # Create Suzanne
    bpy.ops.mesh.primitive_monkey_add(size=2, location=(0, 0, 0))
    suzanne = bpy.context.active_object
    suzanne.name = "Suzanne"
    
    # Add subdivision modifier (will be applied before vertex ID baking)
    subsurf = suzanne.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = SUBDIVISION_LEVEL
    subsurf.render_levels = SUBDIVISION_LEVEL
    
    # Smooth shading for better normals
    bpy.ops.object.shade_smooth()
    
    base_verts = len(suzanne.data.vertices)
    logger.log(f"  Base mesh: {base_verts} vertices")
    logger.log(f"  After subdivision: ~{base_verts * (4 ** SUBDIVISION_LEVEL)} vertices (estimated)")
    
    return suzanne


def create_camera() -> bpy.types.Object:
    """
    Create a fixed camera looking at the origin.
    
    Why Fixed Camera:
        This experiment studies reflectance when only the object rotates.
        The camera position never changes, which is why we need
        correspondence tracking to follow surface points.
    
    Returns:
        The created camera object
    """
    logger.log(f"Creating fixed camera at distance {CAMERA_DISTANCE}...")
    
    # Position camera on negative Y axis, looking at origin
    bpy.ops.object.camera_add(location=(0, -CAMERA_DISTANCE, 0))
    camera = bpy.context.active_object
    camera.name = "Camera"
    
    # Point at origin
    camera.rotation_euler = (math.radians(90), 0, 0)
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    logger.log(f"  Camera position: (0, {-CAMERA_DISTANCE}, 0)")
    logger.log(f"  Camera target: origin (0, 0, 0)")
    
    return camera


def create_light() -> bpy.types.Object:
    """
    Create a directional sun light.
    
    Why Sun Light:
        - No distance falloff (unlike point/spot lights)
        - Matches the simple Lambertian model: I = max(0, n·l)
        - Physically represents distant light source (like the sun)
    
    Why This Direction:
        - (1, 1, 1) normalized creates interesting shading variation
        - Light comes from top-right-front diagonal
        - Creates good contrast between different surface orientations
    
    Returns:
        The created light object
    """
    logger.log("Creating directional sun light...")
    
    # Create sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    light = bpy.context.active_object
    light.name = "Sun"
    
    # Point toward origin
    light.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Set energy
    light.data.energy = 1.0
    
    # Normalize light direction for logging
    light_dir = np.array(LIGHT_DIRECTION)
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    logger.log(f"  Light direction (normalized): ({light_dir[0]:.4f}, {light_dir[1]:.4f}, {light_dir[2]:.4f})")
    
    return light


# ============================================================================
# VERTEX ID BAKING
# ============================================================================

def bake_vertex_ids(obj: bpy.types.Object) -> str:
    """
    Assign unique RGB colors to each vertex based on its index.
    
    The Encoding:
        We use base-256 encoding to pack vertex index into RGB:
        - Red channel:   index % 256         (low byte)
        - Green channel: (index // 256) % 256   (middle byte)
        - Blue channel:  (index // 65536) % 256 (high byte)
        
        This supports up to 256³ = 16.7 million unique vertices.
    
    Why Apply Subdivision First:
        If we bake on the low-poly mesh, interior pixels get interpolated
        colors that don't correspond to real vertex IDs. By applying
        subdivision first, each vertex covers ~1 pixel, minimizing
        interpolation artifacts.
    
    Args:
        obj: The mesh object to bake vertex IDs on
    
    Returns:
        Name of the created vertex color layer
    """
    logger.log("Baking vertex IDs to vertex colors...")
    
    # Ensure object is selected and active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Count base vertices
    base_verts = len(obj.data.vertices)
    logger.log(f"  Base mesh vertices: {base_verts}")
    
    # Apply subdivision modifier BEFORE baking
    # This is critical: we want each subdivided vertex to have its own ID
    logger.log("  Applying subdivision modifier...")
    for mod in obj.modifiers[:]:
        if mod.type == 'SUBSURF':
            bpy.ops.object.modifier_apply(modifier=mod.name)
    
    mesh = obj.data
    final_verts = len(mesh.vertices)
    logger.log(f"  Final mesh vertices: {final_verts}")
    logger.log(f"  Vertex increase: {final_verts - base_verts} new vertices")
    
    # Create or get vertex color layer
    if "VertexID" not in mesh.vertex_colors:
        vcol_layer = mesh.vertex_colors.new(name="VertexID")
    else:
        vcol_layer = mesh.vertex_colors["VertexID"]
    
    # Assign colors based on vertex index
    color_data = vcol_layer.data
    
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            vert_idx = loop.vertex_index
            
            # Encode vertex index as RGB (base-256)
            r = (vert_idx % 256) / 255.0
            g = ((vert_idx // 256) % 256) / 255.0
            b = ((vert_idx // 65536) % 256) / 255.0
            
            color_data[loop_idx].color = (r, g, b, 1.0)
    
    # Log expected color ranges
    max_r = ((final_verts - 1) % 256) / 255.0
    max_g = (((final_verts - 1) // 256) % 256) / 255.0
    max_b = (((final_verts - 1) // 65536) % 256) / 255.0
    
    logger.log(f"  Baked {final_verts} vertex IDs")
    logger.log(f"  Color range - R: [0, {max_r:.3f}], G: [0, {max_g:.3f}], B: [0, {max_b:.3f}]")
    
    if final_verts > 255:
        logger.success("  Vertex count > 255: Green channel will be used")
    if final_verts > 65535:
        logger.success("  Vertex count > 65535: Blue channel will be used")
    
    return vcol_layer.name


# ============================================================================
# MATERIAL CREATION
# ============================================================================

def create_lit_material() -> bpy.types.Material:
    """
    Create material that computes Lambertian shading: I = max(0, n·l).
    
    Why Custom Shader (not Diffuse BSDF):
        - Diffuse BSDF includes energy conservation (1/π factor)
        - Has complex light transport that doesn't match simple n·l
        - We need EXACT n·l values for ground truth comparison
    
    How it Works:
        1. Get surface normal from Geometry node (already world-space)
        2. Create constant light direction vector
        3. Compute dot product: n · l
        4. Clamp to [0, 1]: max(0, n·l)
        5. Output as emission (bypasses all lighting calculations)
    
    Returns:
        The created material
    """
    logger.log("Creating lit material (direct n·l computation)...")
    
    mat = bpy.data.materials.new(name="Lit_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Get surface normal (world space)
    geometry = nodes.new('ShaderNodeNewGeometry')
    geometry.location = (-400, 0)
    
    # Light direction: normalize([1, 1, 1])
    light_dir = np.array(LIGHT_DIRECTION)
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    light_vec = nodes.new('ShaderNodeCombineXYZ')
    light_vec.location = (-400, -200)
    light_vec.inputs['X'].default_value = light_dir[0]
    light_vec.inputs['Y'].default_value = light_dir[1]
    light_vec.inputs['Z'].default_value = light_dir[2]
    
    # Dot product: n · l
    dot = nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    dot.location = (-200, 0)
    
    links.new(geometry.outputs['Normal'], dot.inputs[0])
    links.new(light_vec.outputs['Vector'], dot.inputs[1])
    
    # Clamp to [0, 1]: max(0, n·l)
    clamp = nodes.new('ShaderNodeClamp')
    clamp.location = (0, 0)
    clamp.inputs['Min'].default_value = 0.0
    clamp.inputs['Max'].default_value = 1.0
    
    links.new(dot.outputs['Value'], clamp.inputs['Value'])
    
    # Output as emission (grayscale)
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (200, 0)
    emission.inputs['Strength'].default_value = 1.0
    
    links.new(clamp.outputs['Result'], emission.inputs['Color'])
    
    # Material output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    logger.log(f"  Light direction: ({light_dir[0]:.4f}, {light_dir[1]:.4f}, {light_dir[2]:.4f})")
    logger.log("  Shader: Geometry.Normal -> DotProduct(light) -> Clamp -> Emission")
    
    return mat


def create_vertexid_material(vcol_name: str) -> bpy.types.Material:
    """
    Create material that outputs vertex ID colors as emission.
    
    Why Emission:
        - Emission is not affected by scene lighting
        - The exact vertex color is written to the image
        - No shading, shadows, or reflections to corrupt the ID
    
    Args:
        vcol_name: Name of the vertex color layer to use
    
    Returns:
        The created material
    """
    logger.log("Creating vertex ID material...")
    
    mat = bpy.data.materials.new(name="VertexID_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Vertex color node
    vcol = nodes.new('ShaderNodeVertexColor')
    vcol.location = (0, 0)
    vcol.layer_name = vcol_name
    
    # Emission (unaffected by lighting)
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (200, 0)
    emission.inputs['Strength'].default_value = 1.0
    
    links.new(vcol.outputs['Color'], emission.inputs['Color'])
    
    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    logger.log(f"  Using vertex color layer: '{vcol_name}'")
    
    return mat


def create_normal_material() -> bpy.types.Material:
    """
    Create material that outputs world-space normals as RGB.
    
    Encoding:
        Normal components are in [-1, 1], but images store [0, 1].
        We use: RGB = (normal + 1) / 2
        
        Decoding (in Python later):
        normal = RGB * 2 - 1
    
    Why World-Space:
        - Ground truth computation uses world-space coordinates
        - Light direction is defined in world space
        - Normal changes with object rotation (which is what we're studying)
    
    Important Note:
        Geometry.Normal is ALREADY in world space! We don't need
        any coordinate transformation.
    
    Returns:
        The created material
    """
    logger.log("Creating normal map material (world-space normals)...")
    
    mat = bpy.data.materials.new(name="Normal_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Get normal (already world-space)
    geometry = nodes.new('ShaderNodeNewGeometry')
    geometry.location = (-400, 0)
    
    # Separate XYZ components
    separate = nodes.new('ShaderNodeSeparateXYZ')
    separate.location = (-200, 0)
    links.new(geometry.outputs['Normal'], separate.inputs['Vector'])
    
    # Map each component: [-1, 1] -> [0, 1]
    map_x = nodes.new('ShaderNodeMapRange')
    map_x.location = (0, 100)
    map_x.inputs['From Min'].default_value = -1.0
    map_x.inputs['From Max'].default_value = 1.0
    map_x.inputs['To Min'].default_value = 0.0
    map_x.inputs['To Max'].default_value = 1.0
    links.new(separate.outputs['X'], map_x.inputs['Value'])
    
    map_y = nodes.new('ShaderNodeMapRange')
    map_y.location = (0, 0)
    map_y.inputs['From Min'].default_value = -1.0
    map_y.inputs['From Max'].default_value = 1.0
    map_y.inputs['To Min'].default_value = 0.0
    map_y.inputs['To Max'].default_value = 1.0
    links.new(separate.outputs['Y'], map_y.inputs['Value'])
    
    map_z = nodes.new('ShaderNodeMapRange')
    map_z.location = (0, -100)
    map_z.inputs['From Min'].default_value = -1.0
    map_z.inputs['From Max'].default_value = 1.0
    map_z.inputs['To Min'].default_value = 0.0
    map_z.inputs['To Max'].default_value = 1.0
    links.new(separate.outputs['Z'], map_z.inputs['Value'])
    
    # Combine back to RGB
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (200, 0)
    links.new(map_x.outputs['Result'], combine.inputs['X'])
    links.new(map_y.outputs['Result'], combine.inputs['Y'])
    links.new(map_z.outputs['Result'], combine.inputs['Z'])
    
    # Emission output
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (400, 0)
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    logger.log("  Encoding: RGB = (normal + 1) / 2")
    logger.log("  Coordinate space: World")
    
    return mat


# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def setup_render_settings():
    """
    Configure Blender render settings for EXR output.
    
    Why EXR:
        - 32-bit float precision (no quantization)
        - HDR support for accurate intensity values
        - Lossless compression
        
    Why Cycles:
        - Physically-based renderer
        - Accurate geometry/normal handling
        - Emission shaders work correctly
    """
    logger.log(f"Setting up render settings ({RESOLUTION}x{RESOLUTION})...")
    
    scene = bpy.context.scene
    
    # Resolution
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    
    # Use Cycles for accurate rendering
    scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'  # GPU if available
    bpy.context.scene.cycles.samples = 1  # 1 sample for emission (no noise)
    
    # EXR output
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '32'
    
    # Transparent background
    scene.render.film_transparent = True
    
    # Disable denoising (not needed for emission)
    bpy.context.scene.cycles.use_denoising = False
    
    logger.log(f"  Engine: Cycles")
    logger.log(f"  Format: OpenEXR 32-bit")
    logger.log(f"  Samples: 1 (emission only)")


def rotate_object(obj: bpy.types.Object, angle_deg: float):
    """
    Rotate object around Z-axis.
    
    Why Z-axis:
        - Natural "turntable" rotation
        - Object spins like on a lazy susan
        - Camera sees different sides as it rotates
    
    Args:
        obj: Object to rotate
        angle_deg: Rotation angle in degrees
    """
    obj.rotation_euler = (0, 0, math.radians(angle_deg))


def render_frame(output_path: Path):
    """
    Render current frame to file.
    
    Args:
        output_path: Path to save the rendered image
    """
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


# ============================================================================
# MAIN RENDERING LOOP
# ============================================================================

def main():
    """
    Main rendering pipeline.
    
    Pipeline Steps:
        1. Clear scene and set up objects
        2. Create materials for each render type
        3. For each rotation angle:
           a. Rotate object
           b. Render lit image
           c. Render vertex ID map
           d. Render normal map
        4. Save metadata
    """
    logger.section("FIXED CAMERA REFLECTANCE RENDERING PIPELINE")
    logger.log(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    logger.log(f"Theta samples: {THETA_SAMPLES}")
    logger.log(f"Subdivision level: {SUBDIVISION_LEVEL}")
    logger.log(f"Output: {RENDERS_DIR}")
    
    # Create output directory
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Scene setup
    logger.section("SCENE SETUP")
    clear_scene()
    suzanne = create_suzanne()
    camera = create_camera()
    light = create_light()
    
    # Step 2: Bake vertex IDs
    logger.section("VERTEX ID BAKING")
    vcol_name = bake_vertex_ids(suzanne)
    
    # Step 3: Create materials
    logger.section("MATERIAL CREATION")
    lit_mat = create_lit_material()
    vertexid_mat = create_vertexid_material(vcol_name)
    normal_mat = create_normal_material()
    
    # Assign initial material
    if len(suzanne.data.materials) == 0:
        suzanne.data.materials.append(lit_mat)
    else:
        suzanne.data.materials[0] = lit_mat
    
    # Step 4: Setup render settings
    setup_render_settings()
    
    # Step 5: Render loop
    logger.section("RENDERING")
    logger.log(f"Rendering {THETA_SAMPLES} rotations x 3 image types = {THETA_SAMPLES * 3} images...")
    
    start_time = datetime.now()
    
    for i in range(THETA_SAMPLES):
        theta_deg = (i / THETA_SAMPLES) * 360.0
        
        # Rotate object
        rotate_object(suzanne, theta_deg)
        
        logger.log(f"[{i+1}/{THETA_SAMPLES}] theta = {theta_deg:.1f}°")
        
        # Render lit image
        suzanne.data.materials[0] = lit_mat
        render_frame(RENDERS_DIR / f"theta_{i:03d}.exr")
        
        # Render vertex ID map
        suzanne.data.materials[0] = vertexid_mat
        render_frame(RENDERS_DIR / f"theta_{i:03d}_vertexid.exr")
        
        # Render normal map
        suzanne.data.materials[0] = normal_mat
        render_frame(RENDERS_DIR / f"theta_{i:03d}_normal.exr")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.log(f"Rendering completed in {elapsed:.1f} seconds")
    
    # Step 6: Save metadata
    logger.section("SAVING METADATA")
    
    # Get final vertex count
    final_verts = len(suzanne.data.vertices)
    
    light_dir = np.array(LIGHT_DIRECTION)
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    metadata = {
        "resolution": RESOLUTION,
        "theta_samples": THETA_SAMPLES,
        "subdivision_level": SUBDIVISION_LEVEL,
        "vertex_count": final_verts,
        "camera_distance": CAMERA_DISTANCE,
        "light_direction": light_dir.tolist(),
        "light_direction_raw": list(LIGHT_DIRECTION),
        "render_time_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "files": {
            "lit": [f"theta_{i:03d}.exr" for i in range(THETA_SAMPLES)],
            "vertexid": [f"theta_{i:03d}_vertexid.exr" for i in range(THETA_SAMPLES)],
            "normal": [f"theta_{i:03d}_normal.exr" for i in range(THETA_SAMPLES)]
        }
    }
    
    with open(RENDERS_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.log(f"Saved metadata to {RENDERS_DIR / 'metadata.json'}")
    
    # Summary
    logger.section("RENDERING COMPLETE")
    logger.success(f"Total images: {THETA_SAMPLES * 3}")
    logger.success(f"Output directory: {RENDERS_DIR}")
    logger.success(f"Vertex count: {final_verts}")
    logger.success(f"Render time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

