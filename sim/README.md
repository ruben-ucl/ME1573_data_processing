# RayTracer HDF5 Integration

Complete toolkit for running RayTracer laser absorption analysis on HDF5 keyhole segmentation data.

## Quick Start

### 1. Test Mode (Parameter Tuning)

Test perimeter extraction parameters visually before running full ray tracing:

```bash
# First time setup: Create your config from template
cp sim/test_config_template.json sim/perimeter_extraction_config.json

# Edit sim/perimeter_extraction_config.json to add your track IDs:
# "test_trackids": [
#   "0514_01",
#   "0515_06",
#   "0517_04"
# ]
# Note: File paths are automatically constructed from dirs/hdf5.txt

# Run test mode (uses default config if not specified)
python sim/test_perimeter_extraction.py

# Or specify custom config
python sim/test_perimeter_extraction.py --config sim/my_config.json

# Override parameters on command line
python sim/test_perimeter_extraction.py --opening-width 150 --smoothing 0.7
```

**Outputs** (in `sim/perimeter_extraction_test_results/opening{W}um_smooth{S}_{timestamp}/`):
- Multi-panel figure (3x4 grid showing 12 frames)
- Animated GIF (all middle 50% frames)
- Summary statistics

### 2. Full Pipeline (Production)

Once you're satisfied with extraction parameters, run complete ray tracing:

```bash
# Basic usage (uses default parameters)
python sim/run_raytracer_hdf5.py path/to/track.hdf5

# Custom parameters
python sim/run_raytracer_hdf5.py track.hdf5 \
    --opening-width 150 \
    --smoothing 0.7 \
    --eta 0.2 \
    --n-rays 300 \
    --max-bounces 15

# Process multiple files
python sim/run_raytracer_hdf5.py track1.hdf5 track2.hdf5 track3.hdf5
```

**Outputs** (in `sim/raytracer_results/{trackname}/`):
- `{trackname}_perimeters.csv` - Extracted perimeter coordinates
- `{trackname}_absorptions.csv` - Absorption data per frame
- `{trackname}_absorption_plot.png` - Time series visualization

### 3. CSV Export Only

Export perimeters to CSV without running ray tracing:

```bash
python sim/export_keyhole_to_csv.py input.hdf5 output.csv

# With custom parameters
python sim/export_keyhole_to_csv.py input.hdf5 output.csv \
    --opening-width 150 \
    --smoothing 0.7 \
    --frame-start 100 \
    --frame-end 500
```

## Key Parameters

### Extraction Parameters
- `--opening-width` (default: 120 μm): Width of artificial surface opening
- `--smoothing` (default: 0.5): Spline smoothing factor (0=none, higher=smoother)
- `--resolution` (default: 4.3): Image resolution in μm/pixel

### RayTracer Parameters
- `--eta` (default: 0.175): Absorptivity coefficient
- `--n-rays` (default: 200): Number of rays per frame (more = accurate but slower)
- `--max-bounces` (default: 10): Maximum ray reflections
- `--distribution` (default: gaussian): Ray distribution (gaussian/uniform)
- `--ray-radius` (default: 200 μm): Radius of ray source

## Files

- `hdf5_keyhole_reader.py` - Core perimeter extraction library
- `test_perimeter_extraction.py` - Visual parameter tuning tool
- `export_keyhole_to_csv.py` - Standalone CSV converter
- `run_raytracer_hdf5.py` - Full pipeline script
- `RayTracer.py` - Ray tracing engine (modified for HDF5 compatibility)
- `test_config_template.json` - Template for test configuration
- `perimeter_extraction_config.json` - Default test config (user-created, gitignored)

## Algorithm Details

### Opening Closure
The keyhole surface opening is not resolved in binary data. The extraction algorithm:
1. Finds keyhole center (mean X coordinate)
2. Identifies top-left and top-right endpoints
3. Creates artificial closure:
   - Vertical segments from endpoints to y=0 (top edge)
   - Horizontal segment along top edge
   - Total width: 120 μm (configurable), centered on keyhole
4. Applies cubic spline smoothing to round corners
5. Orders points from top-right to top-left

### Coordinate System
- Origin: Top-left corner (y=0 at top)
- Units: Microns
- No Y-flip needed (unlike legacy CSV data)

## Requirements

**Path Configuration:**
- HDF5 base directory must be defined in `dirs/hdf5.txt`
- Test scripts use track IDs (e.g., `0514_01`), not full paths
- Full paths are constructed as: `{hdf5_base_path}/{trackid}.hdf5`

**HDF5 files must contain:**
- `keyhole_bin` dataset: Binary keyhole masks (3D array: n_frames × height × width)
- `bs-p5-s5` dataset: Background-subtracted raw images (for visualization)

**Development Rule:**
- **No hardcoded paths in scripts** - only in config files and `dirs/*.txt`
- Use `get_paths()` from `tools.py` for centralized path management

## Workflow

```
1. Create test config (one-time setup)
   ↓
2. Run test mode to visually check extraction
   ↓
3. Adjust opening-width and smoothing parameters
   ↓
4. Once satisfied, run full pipeline
   ↓
5. Analyze absorption results
```

## Notes

- All scripts should be run from project root: `python sim/script.py`
- Test mode processes middle 50% of frames (skips first/last 25%)
- Full pipeline processes all frames by default
- RayTracer parameters maintain backward compatibility (legacy data: scale_factor=4.0, flip_y=True)
