#!/bin/bash
################################################################################
# RUN_PIPELINE.SH - Complete Video Relighting Pipeline
################################################################################
#
# This script runs the entire video relighting pipeline:
#   1. Renders 3D object with correspondence tracking data
#   2. Builds correspondence map from vertex IDs
#   3. Extracts reflectance functions and creates 2D maps
#   4. Demonstrates video relighting with multiple lights
#
# Usage:
#   ./run_pipeline.sh              # Run with default settings
#   ./run_pipeline.sh --install    # Install dependencies first
#   ./run_pipeline.sh --high-res   # Use high-resolution settings (for powerful PCs)
#
# Requirements:
#   - Blender 3.0+ (will attempt to locate automatically)
#   - Python 3.8+ with: numpy, matplotlib, imageio
#
# Author: Abel & Team
# Date: January 2026
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

# Default parameters (moderate resolution for testing)
RESOLUTION=512
THETA_SAMPLES=72
SUBDIVISION_LEVEL=3

# High-resolution parameters (for paper-ready results)
HIGH_RES_RESOLUTION=1024
HIGH_RES_THETA_SAMPLES=360
HIGH_RES_SUBDIVISION_LEVEL=4

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

# ============================================================================
# DEPENDENCY DETECTION
# ============================================================================

find_blender() {
    # Try common Blender installation paths
    local blender_paths=(
        "blender"
        "/Applications/Blender.app/Contents/MacOS/Blender"
        "/Applications/Blender.app/Contents/MacOS/blender"
        "$HOME/Applications/Blender.app/Contents/MacOS/Blender"
        "/usr/local/bin/blender"
        "/usr/bin/blender"
        "/snap/bin/blender"
    )
    
    for path in "${blender_paths[@]}"; do
        if command -v "$path" &> /dev/null; then
            echo "$path"
            return 0
        fi
    done
    
    # Check if BLENDER_PATH environment variable is set
    if [ -n "$BLENDER_PATH" ] && [ -x "$BLENDER_PATH" ]; then
        echo "$BLENDER_PATH"
        return 0
    fi
    
    return 1
}

check_python_deps() {
    local missing=()
    
    python3 -c "import numpy" 2>/dev/null || missing+=("numpy")
    python3 -c "import matplotlib" 2>/dev/null || missing+=("matplotlib")
    python3 -c "import imageio" 2>/dev/null || missing+=("imageio")
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo "${missing[@]}"
        return 1
    fi
    return 0
}

# ============================================================================
# INSTALLATION FUNCTIONS
# ============================================================================

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    pip3 install --upgrade pip
    pip3 install numpy matplotlib imageio
    
    log_success "Python dependencies installed!"
}

install_blender_macos() {
    log_info "Attempting to install Blender on macOS..."
    
    if command -v brew &> /dev/null; then
        log_info "Using Homebrew to install Blender..."
        brew install --cask blender
        log_success "Blender installed via Homebrew!"
    else
        log_warning "Homebrew not found. Please install Blender manually:"
        log_warning "  1. Go to https://www.blender.org/download/"
        log_warning "  2. Download Blender for macOS"
        log_warning "  3. Move to /Applications"
        log_warning "  4. Set BLENDER_PATH environment variable if needed"
        return 1
    fi
}

install_blender_linux() {
    log_info "Attempting to install Blender on Linux..."
    
    if command -v apt-get &> /dev/null; then
        log_info "Using apt to install Blender..."
        sudo apt-get update
        sudo apt-get install -y blender
    elif command -v snap &> /dev/null; then
        log_info "Using snap to install Blender..."
        sudo snap install blender --classic
    else
        log_warning "Please install Blender manually from https://www.blender.org/download/"
        return 1
    fi
    
    log_success "Blender installed!"
}

# ============================================================================
# MAIN PIPELINE
# ============================================================================

run_pipeline() {
    local blender_path="$1"
    local resolution="$2"
    local theta_samples="$3"
    local subdivision="$4"
    
    log_section "FIXED CAMERA REFLECTANCE PIPELINE"
    log_info "Resolution: ${resolution}x${resolution}"
    log_info "Theta samples: $theta_samples"
    log_info "Subdivision level: $subdivision"
    log_info "Blender: $blender_path"
    
    # Update render.py configuration
    log_section "STEP 1: RENDERING IN BLENDER"
    log_info "Creating render configuration..."
    
    # Create temporary config override
    cat > "$SCRIPT_DIR/.render_config.py" << EOF
# Auto-generated configuration override
RESOLUTION = $resolution
THETA_SAMPLES = $theta_samples
SUBDIVISION_LEVEL = $subdivision
EOF
    
    log_info "Running Blender render..."
    "$blender_path" --background --python "$SCRIPT_DIR/render.py" \
        2>&1 | tee "$LOGS_DIR/blender_output.log"
    
    if [ $? -eq 0 ]; then
        log_success "Rendering complete!"
    else
        log_error "Rendering failed! Check $LOGS_DIR/blender_output.log"
        return 1
    fi
    
    log_section "STEP 2: BUILDING CORRESPONDENCE MAP"
    log_info "Running correspondence builder..."
    
    python3 "$SCRIPT_DIR/build_correspondence.py" \
        2>&1 | tee -a "$LOGS_DIR/pipeline_output.log"
    
    if [ $? -eq 0 ]; then
        log_success "Correspondence map built!"
    else
        log_error "Correspondence building failed!"
        return 1
    fi
    
    log_section "STEP 3: EXTRACTING REFLECTANCE"
    log_info "Running reflectance extraction..."
    
    python3 "$SCRIPT_DIR/extract_reflectance.py" \
        2>&1 | tee -a "$LOGS_DIR/pipeline_output.log"
    
    if [ $? -eq 0 ]; then
        log_success "Reflectance extraction complete!"
    else
        log_error "Reflectance extraction failed!"
        return 1
    fi
    
    log_section "STEP 4: VIDEO RELIGHTING"
    log_info "Running video relighting demo..."
    
    python3 "$SCRIPT_DIR/relight_video.py" \
        2>&1 | tee -a "$LOGS_DIR/pipeline_output.log"
    
    if [ $? -eq 0 ]; then
        log_success "Video relighting complete!"
    else
        log_error "Video relighting failed!"
        return 1
    fi
    
    # Cleanup temp config
    rm -f "$SCRIPT_DIR/.render_config.py"
    
    log_section "PIPELINE COMPLETE!"
    log_success "All outputs saved to:"
    log_info "  - Renders: $SCRIPT_DIR/renders/"
    log_info "  - Correspondence: $SCRIPT_DIR/correspondence/"
    log_info "  - Output: $SCRIPT_DIR/output/"
    log_info "  - Docs: $SCRIPT_DIR/docs/"
    log_info "  - Logs: $SCRIPT_DIR/logs/"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    local do_install=false
    local high_res=false
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --install)
                do_install=true
                ;;
            --high-res)
                high_res=true
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --install    Install dependencies (Blender, Python packages)"
                echo "  --high-res   Use high-resolution settings (for powerful PCs)"
                echo "  --help       Show this help message"
                echo ""
                echo "Environment Variables:"
                echo "  BLENDER_PATH  Path to Blender executable"
                exit 0
                ;;
        esac
    done
    
    log_section "VIDEO RELIGHTING PIPELINE"
    log_info "Script directory: $SCRIPT_DIR"
    log_info "Date: $(date)"
    
    # Check/install dependencies
    if $do_install; then
        log_section "INSTALLING DEPENDENCIES"
        
        # Install Python dependencies
        install_python_deps
        
        # Install Blender if not found
        if ! find_blender > /dev/null 2>&1; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                install_blender_macos
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                install_blender_linux
            else
                log_error "Unsupported OS. Please install Blender manually."
            fi
        fi
    fi
    
    # Check Python dependencies
    log_info "Checking Python dependencies..."
    missing=$(check_python_deps 2>&1) || true
    if [ -n "$missing" ]; then
        log_warning "Missing Python packages: $missing"
        log_info "Installing missing packages..."
        pip3 install $missing
    fi
    log_success "Python dependencies OK!"
    
    # Find Blender
    log_info "Looking for Blender..."
    blender_path=$(find_blender) || {
        log_error "Blender not found!"
        log_error "Please install Blender or set BLENDER_PATH environment variable."
        log_error "Run with --install to attempt automatic installation."
        exit 1
    }
    log_success "Found Blender: $blender_path"
    
    # Set parameters based on --high-res flag
    if $high_res; then
        log_info "Using HIGH-RESOLUTION settings (paper-ready)"
        run_pipeline "$blender_path" $HIGH_RES_RESOLUTION $HIGH_RES_THETA_SAMPLES $HIGH_RES_SUBDIVISION_LEVEL
    else
        log_info "Using STANDARD settings"
        run_pipeline "$blender_path" $RESOLUTION $THETA_SAMPLES $SUBDIVISION_LEVEL
    fi
}

# Run main function
main "$@"

