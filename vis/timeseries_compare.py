#!/usr/bin/env python
"""
CLI entry point for HDF5 time series comparison analysis (AMPM + KH data).
Analysis pipeline lives in vis/timeseries/. Dataset and processing config in process_single_file().
"""

# Fix KMeans memory leak on Windows - MUST be set before importing sklearn
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from vis.timeseries import DatasetConfig, ProcessingConfig, TimeSeriesComparator
from tools import get_paths
import argparse
import glob

import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def process_single_file(hdf5_file, output_dir=None, verbose=True, run_summary_dir=None,
                        cache_data=False):
    """
    Process a single HDF5 file for time series comparison.

    Parameters:
    -----------
    hdf5_file : str or Path
        Path to the HDF5 file to process
    output_dir : str or Path, optional
        Output directory for results. If None, uses trackid-based subdirectory
    verbose : bool, default=True
        Enable verbose output (prints, debug info)
    run_summary_dir : str or Path, optional
        Directory to copy compact scatter plot for run summary
    cache_data : bool, default=False
        If True, return processed data arrays for batch overview plotting

    Returns:
    --------
    tuple
        (success: bool, summary_row: pd.Series or None)
    """
    import sys
    import io
    
    # Suppress matplotlib display in non-verbose mode
    if not verbose:
        matplotlib.use('Agg')

        # Redirect stdout to suppress print statements
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    # Extract trackid from filename
    trackid = Path(hdf5_file).stem

    # Hardcoded dataset configurations (modify as needed)
    datasets = [
        DatasetConfig(
            group='AMPM',
            name='Photodiode1Bits',
            label='PD1',
            linestyle='-',
            time_group='AMPM',
            time_name='Time',
            time_units='s',
            time_shift=0.0
        ),
        DatasetConfig(
            group='AMPM',
            name='Photodiode2Bits',
            label='PD2',
            linestyle='-',
            time_group='AMPM',
            time_name='Time',
            time_units='s',
            time_shift=0.0
        ),
        # DatasetConfig(
            # group='AMPM',
            # name='BeamDumpDiodeBits',
            # label='BD',
            # linestyle='-',
            # time_group='AMPM',
            # time_name='Time',
            # time_units='s',
            # time_shift=0.0
        # ),
        DatasetConfig(
            group='KH',
            name='max_depth',
            label='KH depth',
            linestyle='-',
            time_group='KH',
            time_name='time',
            time_units='s',
            time_shift=0.0
        ),
        DatasetConfig(
            group='KH',
            name='area',
            label='KH area',
            color='#57106e',
            linestyle='--',
            time_group='KH',
            time_name='time',
            time_units='s',
            time_shift=0.0
        ),
        DatasetConfig(
            group='KH',
            name='fkw_angle',
            label='FKW angle',
            linestyle=':',
            time_group='KH',
            time_name='time',
            time_units='s',
            time_shift=0.0
        ),
        DatasetConfig(
            group='KH',
            name='max_length',
            label='KH length',
            color='#57106e',
            linestyle='-',
            time_group='KH',
            time_name='time',
            time_units='s',
            time_shift=0.0
        )
    ]

    # Processing configuration
    processing_config = ProcessingConfig(
        apply_savgol = False,
        savgol_window = 31,
        savgol_polyorder = 2,
        apply_lowpass = False,
        lowpass_cutoff = 0.4,
        lowpass_order = 4,
        apply_detrend = False,
        detrend_method = 'linear',
        apply_normalization = True,
        normalization_method = 'minmax',
        # Global normalization (optional): Use statistics computed across all tracks
        # To enable, set use_global_normalization=True and run compute_hdf5_statistics.py first
        use_global_normalization = True,  # Default: per-track normalization
        global_stats_file = None,  # Auto-detected from HDF5 directory
        remove_outliers = True,
        outlier_method = ['second_derivative', 'second_derivative'],
        outlier_threshold = [4.5, 4.5],
        outlier_window = 0,
        apply_highpass = False,
        apply_bandpass = False,
        apply_smoothing = False,
        apply_resampling = False,
        sync_method = 'downsample',  # 'downsample' = coarsest grid, 'upsample' = finest grid
        apply_auto_alignment = False,
        alignment_method = 'mutual_info',
        apply_manual_lag_corrections = True  # Set to True to use MANUAL_LAG_CORRECTIONS dict
    )
    
    # Manual lag corrections per track (in seconds)
    # Format: {'trackid': {'group': time_shift_in_seconds}}
    # Positive values delay the signal, negative values advance it
    MANUAL_LAG_CORRECTIONS = {
        '1112_01': {'KH': 0.00162}, 
        '1112_02': {'KH': 0.00162}, 
        '1112_03': {'KH': 0.00162},  
        '1112_04': {'KH': 0.00168},
        '1112_05': {'KH': 0.00168}, 
        '1112_06': {'KH': 0.00168},  
    }

    # Set output directory
    if output_dir is None:
        output_parent = Path(hdf5_file).parent
        output_dir = Path(output_parent, 'timeseries_compare_analysis_results', trackid)
    else:
        output_dir = Path(output_dir)

    try:
        # Initialize comparator
        comparator = TimeSeriesComparator(
            hdf5_path=hdf5_file,
            datasets=datasets,
            processing_config=processing_config
        )

        # Load and process data
        comparator.load_data()
        comparator.process_data()

        # Automatic alignment (optional - controlled by processing_config)
        if processing_config.apply_auto_alignment:
            # Step 1: Compute cross-correlations for ALL signal pairs (only for CCF method)
            if processing_config.alignment_method == 'ccf':
                if verbose:
                    print("\nComputing cross-correlations for all signal pairs...")
                comparator.compute_all_cross_correlations(
                    max_shift_time=processing_config.max_shift_time,
                    correlation_method=processing_config.correlation_method,
                    use_raw_data=processing_config.use_raw_data
                )

            # Step 2: Apply alignment using specified method
            calculated_shifts = comparator.auto_align_time_series(
                reference_label=processing_config.alignment_reference_label,
                reference_group=processing_config.alignment_reference_group,
                correlation_window_time=processing_config.correlation_window_time,
                use_raw_data=processing_config.use_raw_data,
                correlation_method=processing_config.correlation_method,
                max_shift_time=processing_config.max_shift_time,
                sync_within_groups=processing_config.sync_within_groups,
                normalize_signals=processing_config.normalize_signals,
                alignment_method=processing_config.alignment_method,
                feature_method=processing_config.feature_method,
                n_features=processing_config.n_features,
                mi_bins=processing_config.mi_bins,
                use_precomputed_correlations=(processing_config.alignment_method == 'ccf')
            )
            comparator.apply_calculated_shifts(calculated_shifts)

            # Log alignment information with diagnostics (create params dict for logging)
            alignment_params_log = {
                'reference_label': processing_config.alignment_reference_label,
                'reference_group': processing_config.alignment_reference_group,
                'correlation_window_time': processing_config.correlation_window_time,
                'max_shift_time': processing_config.max_shift_time,
                'correlation_method': processing_config.correlation_method,
                'normalize_signals': processing_config.normalize_signals,
                'sync_within_groups': processing_config.sync_within_groups,
                'use_raw_data': processing_config.use_raw_data,
                'alignment_method': processing_config.alignment_method,
                'feature_method': processing_config.feature_method,
                'n_features': processing_config.n_features,
                'mi_bins': processing_config.mi_bins
            }
            comparator.processing_log.add_alignment_info(
                comparator.alignment_info,
                processing_config.apply_auto_alignment,
                alignment_params_log,
                calculated_shifts,
                comparator.alignment_diagnostics if hasattr(comparator, 'alignment_diagnostics') else None
            )

        # Manual lag corrections (optional - overrides auto-alignment)
        if processing_config.apply_manual_lag_corrections and trackid in MANUAL_LAG_CORRECTIONS:
            print(f"\n{'='*80}")
            print(f"APPLYING MANUAL LAG CORRECTIONS FOR {trackid}")
            print(f"{'='*80}")

            manual_shifts = {}
            track_corrections = MANUAL_LAG_CORRECTIONS[trackid]

            # Get dataset groups
            groups = {}
            for dataset_config in comparator.datasets:
                if dataset_config.group not in groups:
                    groups[dataset_config.group] = []
                groups[dataset_config.group].append(dataset_config.label)

            # Apply manual corrections per group
            for group_name, time_shift in track_corrections.items():
                if group_name in groups:
                    print(f"\nApplying {time_shift*1000:.3f}ms shift to {group_name} group:")
                    for label in groups[group_name]:
                        manual_shifts[label] = time_shift
                        print(f"  {label}: {time_shift*1000:+.3f}ms")
                else:
                    print(f"\nWarning: Group '{group_name}' not found in datasets")

            # Apply the manual shifts
            comparator.apply_calculated_shifts(manual_shifts)

            # Update log to reflect manual corrections
            comparator.processing_log.add_alignment_info(
                comparator.alignment_info,
                False,  # Not auto-alignment
                {'manual_corrections': track_corrections},
                manual_shifts,
                None
            )
            print(f"\n✓ Manual lag corrections applied")

        # Debug time shifts (verbose only)
        if verbose:
            print("\n=== Time Vector Debug ===")
            for label in ['PD1', 'PD2', 'KH depth']:
                if label in comparator.time_vectors:
                    current_shift = comparator.alignment_info[label]['time_shift']
                    time_start = comparator.time_vectors[label][0]
                    original_start = comparator.original_time_vectors[label][0]
                    print(f"{label}:")
                    print(f"  Total shift: {current_shift:.6f}s")
                    print(f"  Original start: {original_start:.6f}s")
                    print(f"  Current start: {time_start:.6f}s")
                    print(f"  Expected start: {original_start + current_shift:.6f}s")
                    print(f"  Match: {abs(time_start - (original_start + current_shift)) < 1e-6}")

        # Crop to shortest signal
        cropping_info = comparator.crop_to_shortest_signal(use_processed_data=True)
        comparator.last_cropping_info = cropping_info

        # Log cropping information
        comparator.processing_log.add_cropping_info(cropping_info)

        if verbose:
            print("\nCropping Summary:")
            print(comparator.get_cropping_summary(cropping_info).to_string(index=False))
            print("\nTime Alignment Summary:")
            print(comparator.get_alignment_summary().to_string(index=False))

        # Calculate statistics
        comparator.calculate_statistics()

        # Calculate correlations and silhouette scores for batch summary
        comparator.correlations = comparator.calculate_correlations()
        comparator.silhouette_scores = comparator.calculate_silhouette_scores()

        # Print correlation table
        print(f"\n{'Pair':<35s} {'N':>7s}  {'Pearson r':>10s}  {'Pearson p':>10s}  {'Spearman r':>10s}  {'Spearman p':>10s}")
        print("-" * 95)
        for pair_key, cd in comparator.correlations.items():
            print(f"{pair_key:<35s} {cd['n_actual']:>7d}  {cd['pearson']:>+10.4f}  {cd['pearson_p']:>10.4g}  {cd['spearman']:>+10.4f}  {cd['spearman_p']:>10.4g}")

        if verbose:
            comparator.get_data_summary()

        # Generate comprehensive report
        output_dir.mkdir(parents=True, exist_ok=True)
        comparator.generate_report(output_dir)

        # Generate batch summary row if run_summary_dir is provided
        summary_row = None
        if run_summary_dir is not None:
            run_summary_dir = Path(run_summary_dir)
            run_summary_dir.mkdir(parents=True, exist_ok=True)

            # Copy compact scatter plot to run summary
            compact_scatter_src = Path(output_dir, 'scatterplot_matrix_compact.png')
            compact_scatter_dst = Path(run_summary_dir, f'{trackid}_compact_scatter.png')

            if compact_scatter_src.exists():
                shutil.copy2(compact_scatter_src, compact_scatter_dst)

            # Generate batch summary row
            summary_row = comparator.get_batch_summary_row(trackid)

        if verbose:
            print(f"\n✓ Analysis complete for {trackid}!")
            print(f"✓ Results saved to: {output_dir}")
            if run_summary_dir:
                print(f"✓ Compact scatter plot copied to run summary")
        else:
            # Restore stdout
            sys.stdout = old_stdout

        data_cache = None
        if cache_data:
            data_cache = comparator.get_synchronized_data()

        return True, summary_row, data_cache, comparator if cache_data else None

    except FileNotFoundError:
        if not verbose:
            sys.stdout = old_stdout
        print(f"Error: Could not find HDF5 file at {hdf5_file}")
        return False, None, None, None
    except Exception as e:
        if not verbose:
            sys.stdout = old_stdout
        print(f"Error during analysis of {trackid}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def main():
    """Main function with support for batch and single file processing"""
    parser = argparse.ArgumentParser(
        description='Process HDF5 time series data for comparison analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file (default trackid)
  python vis/timeseries_compare.py

  # Single file with specific trackid
  python vis/timeseries_compare.py --trackid 1112_01

  # Batch: all tracks, no filtering
  python vis/timeseries_compare.py --batch

  # Batch: apply DEFAULT_LOGBOOK_FILTERS from tools.py
  python vis/timeseries_compare.py --batch --filter

  # Batch: defaults + override one field
  python vis/timeseries_compare.py --batch --filter --regime keyhole

  # Batch: single filter in isolation (no defaults loaded)
  python vis/timeseries_compare.py --batch --regime keyhole

  # Batch, silent with progress bar
  python vis/timeseries_compare.py --batch --quiet
        """
    )
    parser.add_argument('--batch', action='store_true',
                       help='Process all .hdf5 files in the data directory')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                       help='Enable verbose output (default: True for single, False for batch)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output (show only progress bar in batch mode)')
    parser.add_argument('--trackid', type=str, default='0105_03',
                       help='Single trackid to process (default: 0105_03)')
    parser.add_argument('--filter', action='store_true',
                       help='Load DEFAULT_LOGBOOK_FILTERS from tools.py as the filter base')
    parser.add_argument('--material', type=str, help='Filter by material (e.g. AlSi10Mg)')
    parser.add_argument('--layer', type=int, help='Filter by layer number')
    parser.add_argument('--laser_mode', type=str, help='Filter by laser mode (cw, pw)')
    parser.add_argument('--base_type', type=str, help='Filter by base type (powder, weld, solid)')
    parser.add_argument('--regime', type=str, help='Filter by melting regime')
    parser.add_argument('--beamtime', type=str, help='Filter by beamtime number(s), comma-separated')
    args = parser.parse_args()

    # Determine verbosity
    if args.batch:
        verbose = args.verbose and not args.quiet  # Default False for batch
    else:
        verbose = not args.quiet  # Default True for single

    folder = get_paths()['hdf5']

    if args.batch:
        # Batch mode: process all .hdf5 files
        hdf5_files = sorted(glob.glob(str(Path(folder) / '*.hdf5')))

        from tools import get_excluded_trackids
        excluded = set(get_excluded_trackids())
        hdf5_files = [f for f in hdf5_files if Path(f).stem not in excluded]

        cli_filter_args = {
            'material': args.material,
            'layer': args.layer,
            'laser_mode': args.laser_mode,
            'base_type': args.base_type,
            'regime': args.regime,
            'beamtime': args.beamtime,
        }
        has_cli_filters = any(v is not None for v in cli_filter_args.values())

        if args.filter or has_cli_filters:
            from tools import get_logbook, filter_logbook_tracks, DEFAULT_LOGBOOK_FILTERS
            filters = DEFAULT_LOGBOOK_FILTERS.copy() if args.filter else {}
            if args.material:          filters['material']   = args.material
            if args.layer is not None: filters['layer']      = args.layer
            if args.laser_mode:        filters['laser_mode'] = args.laser_mode
            if args.base_type:         filters['base_type']  = args.base_type
            if args.regime:            filters['regime']     = args.regime
            if args.beamtime:
                bt = args.beamtime.strip()
                filters['beamtime'] = [int(x) for x in bt.split(',')] if ',' in bt else int(bt)
            logbook = get_logbook()
            filtered_lb, active_filters = filter_logbook_tracks(logbook, filters)
            valid_trackids = set(filtered_lb['trackid'].dropna().astype(str))
            hdf5_files = [f for f in hdf5_files if Path(f).stem in valid_trackids]
            print(f"Logbook filter applied: {active_filters}")
            print(f"  {len(hdf5_files)} tracks pass filter")

        if not hdf5_files:
            print(f"No .hdf5 files found in {folder}")
            return

        # Create run summary directory
        run_summary_dir = Path(folder, 'timeseries_compare_analysis_results', 'run_summary')

        print(f"Processing {len(hdf5_files)} files in batch mode...")
        if not verbose:
            print(f"Output directory: {Path(folder, 'timeseries_compare_analysis_results')}")
            print(f"Run summary directory: {run_summary_dir}")

        success_count = 0
        failed_files = []

        # Initialize CSV path - always start fresh each batch run
        csv_path = Path(run_summary_dir, 'batch_summary.csv')
        if csv_path.exists():
            csv_path.unlink()
        write_header = True

        # Signal cache for batch scatter plot {label: [array_per_track, ...]}
        signal_cache = {}
        n_cached_tracks = 0
        last_comparator = None

        # Use tqdm progress bar in quiet mode
        if not verbose:
            iterator = tqdm(hdf5_files, desc="Processing files", unit="file")
        else:
            iterator = hdf5_files

        for hdf5_file in iterator:
            trackid = Path(hdf5_file).stem
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing {trackid}...")
                print(f"{'='*60}")

            success, summary_row, data_cache, comparator = process_single_file(
                hdf5_file,
                verbose=verbose,
                run_summary_dir=run_summary_dir,
                cache_data=True,
            )

            if success:
                success_count += 1
                if summary_row is not None:
                    df_row = pd.DataFrame([summary_row])
                    df_row.to_csv(csv_path, mode='a', header=write_header, index=False)
                    write_header = False

                # Accumulate synchronized data for batch plots.
                if data_cache:
                    for label, data in data_cache.items():
                        signal_cache.setdefault(label, []).append(data)
                    n_cached_tracks += 1
                    last_comparator = comparator
            else:
                failed_files.append(trackid)
                if verbose:
                    print(f"FAILED: {trackid}")

        # Generate batch plots and stats from concatenated signals
        if signal_cache and n_cached_tracks > 0 and last_comparator is not None:
            print("\nGenerating batch plots...")
            concat_data = {label: np.concatenate(arrays)
                           for label, arrays in signal_cache.items()}
            last_comparator.plot_scatterplot_matrix_compact(
                data=concat_data,
                save_path=run_summary_dir / 'batch_scatter.png',
                max_points=10000, point_size=1.5, point_alpha=0.2,
                title=f'Batch scatter — {n_cached_tracks} tracks',
            )
            last_comparator.plot_scatterplot_matrix_compact(
                data=concat_data,
                save_path=run_summary_dir / 'batch_scatter_correlation.png',
                max_points=10000, point_size=1.5, point_alpha=0.2,
                title=f'Batch scatter — {n_cached_tracks} tracks',
                style='correlation',
            )
            last_comparator.plot_correlation_matrix(
                data=concat_data,
                save_path=run_summary_dir / 'batch_correlation_matrix.png',
                title=f'Batch correlation — {n_cached_tracks} tracks',
            )

            # Write batch-level stats row to the summary CSV
            batch_row = {'trackid': 'BATCH_ALL', 'n_tracks': n_cached_tracks}
            labels_list = list(concat_data.keys())
            for i, l1 in enumerate(labels_list):
                for j, l2 in enumerate(labels_list):
                    if i < j:
                        pair_clean = f'{l1} vs {l2}'.replace(' vs ', '_vs_').replace(' ', '_')
                        r, p = pearsonr(concat_data[l1], concat_data[l2])
                        rho, _ = spearmanr(concat_data[l1], concat_data[l2])
                        batch_row[f'{pair_clean}_pearson'] = round(r, 4)
                        batch_row[f'{pair_clean}_pearson_p'] = round(p, 4)
                        batch_row[f'{pair_clean}_spearman'] = round(rho, 4)
                        batch_row[f'{pair_clean}_N'] = len(concat_data[l1])
            if csv_path.exists():
                existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                batch_df = pd.DataFrame([batch_row]).reindex(columns=existing_cols)
                batch_df.to_csv(csv_path, mode='a', header=False, index=False)
            print(f"Batch stats row appended to {csv_path}")

        # Final summary
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"  Successful: {success_count}/{len(hdf5_files)}")
        if failed_files:
            print(f"  Failed: {len(failed_files)}")
            print(f"  Failed files: {', '.join(failed_files)}")
        print(f"  Results: {Path(folder, 'timeseries_compare_analysis_results')}")
        print(f"  Run summary: {run_summary_dir}")
        if csv_path:
            print(f"  Batch summary CSV: {csv_path}")
        print(f"{'='*60}")

    else:
        # Single file mode
        hdf5_file = Path(folder, args.trackid + '.hdf5')

        if verbose:
            print(f"Processing single file: {args.trackid}")
            print(f"File path: {hdf5_file}")

        success, _, _, _ = process_single_file(str(hdf5_file), verbose=verbose)

        if not success:
            print(f"Failed to process {args.trackid}")
        elif verbose:
            print("\n✓ Analysis complete! Check the 'timeseries_compare_analysis_results' folder for outputs.")
            print("✓ Alignment comparison plot shows original vs shifted time series.")
            print("✓ Cropping summary shows data reduction details.")


if __name__ == '__main__':
    main()