"""
RayTracer HDF5 Pipeline

Complete automation pipeline: HDF5 → Perimeter Extraction → CSV → RayTracer Analysis

This script:
1. Extracts keyhole perimeters from HDF5 binary images
2. Exports to CSV format
3. Runs RayTracer analysis
4. Generates plots and statistics

Usage:
    python run_raytracer_hdf5.py track.hdf5
    python run_raytracer_hdf5.py track.hdf5 --opening-width 150 --eta 0.2

Author: Claude
Date: 2025-11-20
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add sim directory to path for imports when run from project root
sys.path.insert(0, str(Path(__file__).parent))

# Import extraction modules
from hdf5_keyhole_reader import extract_all_perimeters

# Import RayTracer
try:
    from RayTracer import RayTracer
except ImportError:
    print("Error: RayTracer.py not found. Make sure it's in the sim/ directory.")
    sys.exit(1)


def run_raytracer_pipeline(hdf5_path: str,
                           output_dir: Path = None,
                           opening_width_um: float = 120.0,
                           smoothing_factor: float = 0.5,
                           resolution_um_per_pixel: float = 4.3,
                           dataset_name: str = 'keyhole_bin',
                           frame_range: tuple = None,
                           raytracer_params: dict = None):
    """
    Run complete RayTracer analysis pipeline on HDF5 data.

    Parameters:
        hdf5_path: Path to HDF5 file
        output_dir: Output directory (default: sim/raytracer_results/{trackname}/)
        opening_width_um: Keyhole opening width in microns
        smoothing_factor: Spline smoothing factor
        resolution_um_per_pixel: Spatial resolution
        dataset_name: HDF5 dataset name
        frame_range: Optional (start, end) frame range
        raytracer_params: Dictionary of RayTracer parameters

    Returns:
        results: Dictionary with analysis results
    """
    hdf5_path = Path(hdf5_path)
    track_name = hdf5_path.stem

    # Set up output directory
    if output_dir is None:
        output_dir = Path('sim/raytracer_results') / track_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"="*60)
    print(f"RayTracer HDF5 Pipeline")
    print(f"="*60)
    print(f"Track: {track_name}")
    print(f"Input: {hdf5_path}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Extract perimeters
    print(f"[1/3] Extracting keyhole perimeters...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Opening width: {opening_width_um:.1f} μm")
    print(f"  Smoothing: {smoothing_factor:.2f}")
    print(f"  Resolution: {resolution_um_per_pixel:.2f} μm/pixel")

    try:
        df = extract_all_perimeters(
            str(hdf5_path),
            dataset_name=dataset_name,
            opening_width_um=opening_width_um,
            smoothing_factor=smoothing_factor,
            resolution_um_per_pixel=resolution_um_per_pixel,
            frame_range=frame_range
        )

        print(f"  ✓ Extracted {len(df):,} points from {df['Slice'].nunique()} frames")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

    # Step 2: Export to CSV
    print(f"\n[2/3] Exporting to CSV...")
    csv_path = output_dir / f'{track_name}_perimeters.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")

    # Step 3: Run RayTracer analysis
    print(f"\n[3/3] Running RayTracer analysis...")

    if raytracer_params is None:
        raytracer_params = {}

    # Default RayTracer parameters
    eta = raytracer_params.get('eta', 0.175)
    n_rays = raytracer_params.get('n_rays', 200)
    max_bounces = raytracer_params.get('max_bounces', 10)
    ray_distribution = raytracer_params.get('distribution', 'gaussian')
    ray_radius = raytracer_params.get('ray_radius', 200)
    curved = raytracer_params.get('curved', True)

    print(f"  Parameters:")
    print(f"    η (absorptivity): {eta:.3f}")
    print(f"    Rays per slice: {n_rays}")
    print(f"    Max bounces: {max_bounces}")
    print(f"    Distribution: {ray_distribution}")
    print(f"    Curved surface: {curved}")

    try:
        # Initialize RayTracer
        tracer = RayTracer(
            output_directory=str(output_dir),
            path=str(csv_path),
            file_name=track_name,
            eta=eta,
            scale_factor=1.0,  # Data already in microns
            flip_y=False        # No Y-flip needed
        )

        # Get list of slices
        slices = sorted(df['Slice'].unique())
        n_slices = len(slices)

        print(f"  Processing {n_slices} slices...")

        # Process each slice
        absorptions = []
        bounce_counts_all = []

        for i, slice_num in enumerate(slices):
            if (i+1) % 10 == 0 or i == 0 or (i+1) == n_slices:
                print(f"    Slice {i+1}/{n_slices} (frame {slice_num})", end='\r')

            # Load data for this slice
            tracer.load_data(tracer.path, slice=slice_num)

            # Construct polygon
            tracer.construct_polygon(curved=curved)

            # Trace rays
            paths, absorption, bounce_counts = tracer.trace_rays(
                n_rays=n_rays,
                distribution=ray_distribution,
                ray_radius=ray_radius,
                max_bounces=max_bounces
            )

            absorptions.append(absorption)
            bounce_counts_all.append(bounce_counts)

        print(f"\n  ✓ Analysis complete")

        # Calculate statistics
        absorptions = np.array(absorptions)
        mean_absorption = np.mean(absorptions)
        std_absorption = np.std(absorptions)
        min_absorption = np.min(absorptions)
        max_absorption = np.max(absorptions)

        print(f"\n  Absorption Statistics:")
        print(f"    Mean: {mean_absorption:.4f}")
        print(f"    Std:  {std_absorption:.4f}")
        print(f"    Min:  {min_absorption:.4f}")
        print(f"    Max:  {max_absorption:.4f}")

        # Save absorption results
        absorption_path = output_dir / f'{track_name}_absorptions.csv'
        absorption_df = pd.DataFrame({
            'Slice': slices,
            'Absorption': absorptions
        })
        absorption_df.to_csv(absorption_path, index=False)
        print(f"\n  ✓ Saved absorption data: {absorption_path.name}")

        # Generate summary plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(slices, absorptions, 'b-', linewidth=2, label='Absorption')
        ax.axhline(mean_absorption, color='r', linestyle='--', label=f'Mean: {mean_absorption:.4f}')
        ax.fill_between(slices, mean_absorption - std_absorption, mean_absorption + std_absorption,
                        alpha=0.3, color='r', label=f'±1σ: {std_absorption:.4f}')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Absorption', fontsize=12)
        ax.set_title(f'{track_name} - Ray Tracing Absorption Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = output_dir / f'{track_name}_absorption_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved plot: {plot_path.name}")

        results = {
            'success': True,
            'track_name': track_name,
            'n_slices': n_slices,
            'mean_absorption': mean_absorption,
            'std_absorption': std_absorption,
            'min_absorption': min_absorption,
            'max_absorption': max_absorption,
            'output_dir': str(output_dir),
            'csv_path': str(csv_path),
            'absorption_path': str(absorption_path),
            'plot_path': str(plot_path)
        }

        return results

    except Exception as e:
        print(f"  ✗ Error during RayTracer analysis: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='Run RayTracer analysis on HDF5 keyhole data'
    )
    parser.add_argument('hdf5_files', nargs='+', help='HDF5 file(s) to process')
    parser.add_argument('--output-dir', help='Output directory (default: sim/raytracer_results/{trackname}/)')
    parser.add_argument('--dataset', default='keyhole_bin', help='HDF5 dataset name (default: keyhole_bin)')

    # Extraction parameters
    parser.add_argument('--opening-width', type=float, default=120.0,
                       help='Opening width in microns (default: 120.0)')
    parser.add_argument('--smoothing', type=float, default=0.5,
                       help='Smoothing factor (default: 0.5)')
    parser.add_argument('--resolution', type=float, default=4.3,
                       help='Resolution in μm/pixel (default: 4.3)')

    # Frame range
    parser.add_argument('--frame-start', type=int, help='Start frame index (optional)')
    parser.add_argument('--frame-end', type=int, help='End frame index (optional)')

    # RayTracer parameters
    parser.add_argument('--eta', type=float, default=0.175,
                       help='Absorptivity coefficient (default: 0.175)')
    parser.add_argument('--n-rays', type=int, default=200,
                       help='Number of rays per slice (default: 200)')
    parser.add_argument('--max-bounces', type=int, default=10,
                       help='Maximum reflections (default: 10)')
    parser.add_argument('--distribution', choices=['gaussian', 'uniform'], default='gaussian',
                       help='Ray distribution (default: gaussian)')
    parser.add_argument('--ray-radius', type=float, default=200.0,
                       help='Ray radius in microns (default: 200)')
    parser.add_argument('--no-curved', action='store_true',
                       help='Use piecewise linear surface instead of cubic spline')

    args = parser.parse_args()

    # Determine frame range
    frame_range = None
    if args.frame_start is not None or args.frame_end is not None:
        start = args.frame_start if args.frame_start is not None else 0
        end = args.frame_end if args.frame_end is not None else 999999
        frame_range = (start, end)

    # RayTracer parameters
    raytracer_params = {
        'eta': args.eta,
        'n_rays': args.n_rays,
        'max_bounces': args.max_bounces,
        'distribution': args.distribution,
        'ray_radius': args.ray_radius,
        'curved': not args.no_curved
    }

    # Process each file
    all_results = []

    for hdf5_file in args.hdf5_files:
        hdf5_path = Path(hdf5_file)

        if not hdf5_path.exists():
            print(f"Warning: File not found: {hdf5_path}")
            continue

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = None  # Will be set to default in pipeline function

        # Run pipeline
        result = run_raytracer_pipeline(
            str(hdf5_path),
            output_dir=output_dir,
            opening_width_um=args.opening_width,
            smoothing_factor=args.smoothing,
            resolution_um_per_pixel=args.resolution,
            dataset_name=args.dataset,
            frame_range=frame_range,
            raytracer_params=raytracer_params
        )

        if result['success']:
            all_results.append(result)
            print(f"\n{'='*60}")
            print(f"✓ Completed: {result['track_name']}")
            print(f"  Results saved to: {result['output_dir']}")
            print(f"{'='*60}\n")

    # Summary
    if all_results:
        print(f"\nPipeline Summary:")
        print(f"  Tracks processed: {len(all_results)}")
        print(f"  Total slices: {sum(r['n_slices'] for r in all_results)}")
        print(f"\nAbsorption Summary:")
        for r in all_results:
            print(f"  {r['track_name']}: {r['mean_absorption']:.4f} ± {r['std_absorption']:.4f}")


if __name__ == "__main__":
    main()
