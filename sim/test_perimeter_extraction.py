"""
Perimeter Extraction Test Mode

Rapid parameter tuning and visualization tool for keyhole perimeter extraction.
Processes test HDF5 files and generates multi-panel figures and animated GIFs
for quick visual feedback without running full ray tracing.

Usage:
    python test_perimeter_extraction.py --config test_config.json
    python test_perimeter_extraction.py --config test_config.json --opening-width 150

Author: Claude
Date: 2025-11-20
"""

import argparse
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import imageio
from tqdm import tqdm
import sys

# Add project root and sim directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # Project root for tools.py
sys.path.insert(0, str(Path(__file__).parent))          # sim/ for local modules

from hdf5_keyhole_reader import (
    read_keyhole_images,
    extract_keyhole_perimeter_with_opening
)

# Import centralized path management
from tools import get_paths


def load_config(config_path: str) -> Dict:
    """Load test configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_frame_range_middle_50percent(n_frames: int, trim_start_percent: float = 25,
                                     trim_end_percent: float = 25) -> Tuple[int, int]:
    """
    Calculate frame range for middle 50% of data.

    Parameters:
        n_frames: Total number of frames
        trim_start_percent: Percentage to trim from start (default: 25)
        trim_end_percent: Percentage to trim from end (default: 25)

    Returns:
        start_frame, end_frame: Frame indices
    """
    start_frame = int(n_frames * trim_start_percent / 100)
    end_frame = int(n_frames * (100 - trim_end_percent) / 100)
    return start_frame, end_frame


def load_raw_image(hdf5_path: str, frame_idx: int, dataset_name: str = 'bs-p5-s5') -> np.ndarray:
    """Load raw background-subtracted image for visualization."""
    with h5py.File(hdf5_path, 'r') as f:
        if dataset_name in f:
            img = f[dataset_name][frame_idx]
        else:
            # Fallback to zeros if dataset not found
            print(f"Warning: {dataset_name} not found, using blank background")
            # Try to get shape from keyhole_bin
            if 'keyhole_bin' in f:
                shape = f['keyhole_bin'][frame_idx].shape
                img = np.zeros(shape, dtype=np.uint8)
            else:
                img = np.zeros((256, 512), dtype=np.uint8)
    return img


def create_visualization_frame(raw_image: np.ndarray,
                               x_coords: np.ndarray,
                               y_coords: np.ndarray,
                               frame_number: int,
                               resolution: float = 4.3,
                               viz_config: Dict = None) -> np.ndarray:
    """
    Create a single visualization frame with perimeter overlay.

    Parameters:
        raw_image: Background image
        x_coords: X coordinates in microns
        y_coords: Y coordinates in microns
        frame_number: Frame number for annotation
        resolution: Resolution in um/pixel
        viz_config: Visualization configuration dict

    Returns:
        frame: RGB image array
    """
    if viz_config is None:
        viz_config = {}

    # Default visualization settings
    perimeter_color = viz_config.get('perimeter_color', 'cyan')
    perimeter_linewidth = viz_config.get('perimeter_linewidth', 2)
    start_point_color = viz_config.get('start_point_color', 'lime')
    end_point_color = viz_config.get('end_point_color', 'red')
    marker_size = viz_config.get('marker_size', 50)
    desaturate = viz_config.get('desaturate_background', True)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display raw image (desaturated)
    if desaturate:
        # Convert to grayscale-like appearance with reduced contrast
        img_display = raw_image.astype(float)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-10)
        img_display = img_display * 0.5  # Reduce intensity
        ax.imshow(img_display, cmap='gray', alpha=0.7)
    else:
        ax.imshow(raw_image, cmap='gray')

    # Convert coordinates from microns to pixels for plotting
    x_pixels = x_coords / resolution
    y_pixels = y_coords / resolution

    # Plot perimeter
    ax.plot(x_pixels, y_pixels, color=perimeter_color, linewidth=perimeter_linewidth,
            label='Perimeter')

    # Mark start point (top-right)
    ax.scatter(x_pixels[0], y_pixels[0], s=marker_size, color=start_point_color,
              marker='o', edgecolors='white', linewidths=1.5, label='START', zorder=10)
    ax.text(x_pixels[0], y_pixels[0] - 10, 'START', color='white',
            fontsize=8, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=start_point_color, alpha=0.7))

    # Mark end point (top-left)
    ax.scatter(x_pixels[-1], y_pixels[-1], s=marker_size, color=end_point_color,
              marker='o', edgecolors='white', linewidths=1.5, label='END', zorder=10)
    ax.text(x_pixels[-1], y_pixels[-1] - 10, 'END', color='white',
            fontsize=8, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=end_point_color, alpha=0.7))

    # Add frame number
    ax.text(0.02, 0.98, f'Frame {frame_number}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Set aspect and remove axes
    ax.set_aspect('equal')
    ax.axis('off')

    # Convert figure to array
    fig.canvas.draw()
    frame_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame_array = frame_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return frame_array


def process_test_file(hdf5_path: str,
                      parameters: Dict,
                      sampling_config: Dict,
                      viz_config: Dict,
                      output_dir: Path) -> Dict:
    """
    Process a single test file and generate outputs.

    Returns:
        results: Dictionary with processing statistics
    """
    hdf5_path = Path(hdf5_path)
    track_name = hdf5_path.stem

    print(f"\nProcessing {track_name}.hdf5")

    # Load binary images
    dataset_name = parameters.get('dataset_name', 'keyhole_bin')
    raw_dataset_name = parameters.get('raw_image_dataset', 'bs-p5-s5')

    print(f"  Loading datasets: {dataset_name}, {raw_dataset_name}")

    try:
        binary_images, metadata = read_keyhole_images(hdf5_path, dataset_name)
    except Exception as e:
        print(f"  ❌ Error loading {dataset_name}: {e}")
        return {'success': False, 'error': str(e)}

    n_frames = metadata['n_frames']

    # Determine frame range
    start_frame, end_frame = get_frame_range_middle_50percent(
        n_frames,
        sampling_config.get('trim_start_percent', 25),
        sampling_config.get('trim_end_percent', 25)
    )

    selected_frames = list(range(start_frame, end_frame))
    n_selected = len(selected_frames)

    print(f"  Total frames: {n_frames}")
    print(f"  Selected frames: {n_selected} (frames {start_frame}-{end_frame}, middle 50%)")

    # Extract perimeters
    opening_width = parameters['opening_width_um']
    smoothing = parameters['smoothing_factor']
    resolution = parameters['resolution_um_per_pixel']

    print(f"  Extracting perimeters... ", end='', flush=True)

    perimeters = []
    vis_frames = []
    success_count = 0

    for frame_idx in tqdm(selected_frames, desc="  Progress"):
        try:
            # Extract perimeter
            binary_image = binary_images[frame_idx]
            x_coords, y_coords = extract_keyhole_perimeter_with_opening(
                binary_image, opening_width, smoothing, resolution
            )

            perimeters.append({
                'frame': frame_idx,
                'x': x_coords,
                'y': y_coords,
                'length': len(x_coords)
            })

            # Load raw image for visualization
            raw_image = load_raw_image(hdf5_path, frame_idx, raw_dataset_name)

            # Create visualization frame
            vis_frame = create_visualization_frame(
                raw_image, x_coords, y_coords, frame_idx, resolution, viz_config
            )
            vis_frames.append(vis_frame)

            success_count += 1

        except Exception as e:
            print(f"\n  Warning: Frame {frame_idx} failed: {e}")
            continue

    print(f"\n  ✓ Success rate: {success_count}/{n_selected} ({100*success_count/n_selected:.1f}%)")

    if success_count == 0:
        return {'success': False, 'error': 'No frames processed successfully'}

    # Calculate statistics
    perimeter_lengths = [p['length'] for p in perimeters]
    avg_length = np.mean(perimeter_lengths)

    print(f"  ✓ Average perimeter length: {avg_length:.0f} points")

    # Generate multi-panel figure
    if viz_config.get('save_figure', True):
        print(f"  Generating summary figure... ", end='', flush=True)

        # Select frames for figure (evenly spaced, max 12)
        layout = viz_config.get('figure_layout', '3x4')
        n_rows, n_cols = map(int, layout.split('x'))
        n_panels = n_rows * n_cols

        if len(vis_frames) > n_panels:
            indices = np.linspace(0, len(vis_frames)-1, n_panels, dtype=int)
            selected_vis_frames = [vis_frames[i] for i in indices]
        else:
            selected_vis_frames = vis_frames

        fig_path = output_dir / f'{track_name}_summary_figure.png'
        create_multi_panel_figure(selected_vis_frames, fig_path,
                                  track_name, parameters, viz_config)
        print(f"✓ Saved: {fig_path.name}")

    # Generate animated GIF
    if viz_config.get('save_gif', True):
        print(f"  Generating animation... ", end='', flush=True)

        gif_path = output_dir / f'{track_name}_perimeter_animation.gif'
        fps = viz_config.get('gif_fps', 10)

        imageio.mimsave(gif_path, vis_frames, fps=fps)
        print(f"✓ Saved: {gif_path.name}")

    # Processing time
    results = {
        'success': True,
        'track_name': track_name,
        'n_frames_total': n_frames,
        'n_frames_processed': success_count,
        'success_rate': success_count / n_selected,
        'avg_perimeter_length': avg_length
    }

    return results


def create_multi_panel_figure(frames: List[np.ndarray],
                               output_path: Path,
                               track_name: str,
                               parameters: Dict,
                               viz_config: Dict):
    """Create multi-panel summary figure."""
    layout = viz_config.get('figure_layout', '3x4')
    n_rows, n_cols = map(int, layout.split('x'))
    dpi = viz_config.get('figure_dpi', 150)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame)
        ax.axis('off')

    # Hide unused subplots
    for i in range(len(frames), len(axes)):
        axes[i].axis('off')

    # Add title
    title = f"{track_name} - Opening: {parameters['opening_width_um']:.0f} μm, Smoothing: {parameters['smoothing_factor']:.2f}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test perimeter extraction with visualization')
    parser.add_argument('--config', default='sim/perimeter_extraction_config.json',
                       help='Path to test configuration JSON file (default: sim/perimeter_extraction_config.json)')
    parser.add_argument('--opening-width', type=float, help='Override opening width (μm)')
    parser.add_argument('--smoothing', type=float, help='Override smoothing factor')

    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        print(f"Please create a config file or use --config to specify an existing one.")
        print(f"You can use sim/test_config_template.json as a starting point.")
        sys.exit(1)

    config = load_config(config_path)

    # Override parameters if specified
    if args.opening_width is not None:
        config['parameters']['opening_width_um'] = args.opening_width
    if args.smoothing is not None:
        config['parameters']['smoothing_factor'] = args.smoothing

    # Create output directory with descriptive name
    output_base = Path(config['output']['output_dir'])
    opening_width = config['parameters']['opening_width_um']
    smoothing = config['parameters']['smoothing_factor']
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    output_dir = output_base / f"opening{opening_width:.0f}um_smooth{smoothing:.1f}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running perimeter extraction test...")
    print(f"Config: {args.config}")
    print(f"Output folder: {output_dir}")

    # Save configuration used
    config_out_path = output_dir / 'config_used.json'
    with open(config_out_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Get HDF5 base path from centralized configuration
    paths = get_paths()
    if 'hdf5' not in paths:
        print(f"Error: HDF5 base path not configured.")
        print(f"Please define the HDF5 directory in dirs/hdf5.txt")
        sys.exit(1)

    hdf5_base_path = paths['hdf5']
    print(f"HDF5 base directory: {hdf5_base_path}")
    print()

    # Process each test file
    all_results = []

    for trackid in config['test_trackids']:
        if trackid.startswith('#'):
            continue  # Skip comments

        # Construct full path from trackid
        hdf5_path = hdf5_base_path / f'{trackid}.hdf5'

        result = process_test_file(
            hdf5_path,
            config['parameters'],
            config['sampling'],
            config['visualization'],
            output_dir
        )

        if result['success']:
            all_results.append(result)

    # Generate summary
    print(f"\nTest complete!")
    print(f"Total tracks: {len(all_results)}")

    if all_results:
        total_frames = sum(r['n_frames_processed'] for r in all_results)
        avg_success = np.mean([r['success_rate'] for r in all_results])

        print(f"Total frames processed: {total_frames}")
        print(f"Average success rate: {avg_success*100:.1f}%")

        # Save summary
        summary_path = output_dir / 'test_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Perimeter Extraction Test Summary\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Opening width: {opening_width:.1f} μm\n")
            f.write(f"Smoothing factor: {smoothing:.2f}\n")
            f.write(f"Resolution: {config['parameters']['resolution_um_per_pixel']:.2f} μm/pixel\n\n")
            f.write(f"Results:\n")
            f.write(f"  Tracks processed: {len(all_results)}\n")
            f.write(f"  Total frames: {total_frames}\n")
            f.write(f"  Success rate: {avg_success*100:.1f}%\n\n")

            for r in all_results:
                f.write(f"\n{r['track_name']}:\n")
                f.write(f"  Frames processed: {r['n_frames_processed']}/{r['n_frames_total']}\n")
                f.write(f"  Success rate: {r['success_rate']*100:.1f}%\n")
                f.write(f"  Avg perimeter length: {r['avg_perimeter_length']:.0f} points\n")

    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
