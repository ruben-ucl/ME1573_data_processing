"""
Standalone script for visualizing track-level predictions from saved model outputs.

This script can operate in two modes:
1. Load saved predictions from test_predictions_vXXX.pkl (fast)
2. Regenerate predictions from saved model and test data (fallback)

Usage:
    python ml/visualize_track_predictions.py --version v001 --classifier_type cwt_image
    python ml/visualize_track_predictions.py --version v115 --dataset_variant AlSi10Mg_CW_L1_powder_porosity_with-test_auto-split
    python ml/visualize_track_predictions.py --version v001 --force-regenerate
"""

import os
import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# Set UTF-8 encoding for all I/O operations
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.config import get_model_dir, format_version

# ============================================================================
# COLOR SCHEME CONFIGURATION
# ============================================================================
# Centralized color definitions for easy customization
COLOR_NO_POROSITY = '#3498db'      # Blue for no porosity (class 0)
COLOR_POROSITY = '#e74c3c'         # Red for porosity (class 1)
COLOR_SKIPPED_WINDOW = '#95a5a6'   # Grey for skipped first window
# ============================================================================


def load_saved_predictions(version, model_dir):
    """Load predictions from saved pickle file."""
    predictions_file = model_dir / 'test_evaluation' / f'test_predictions_{version}.pkl'

    if not predictions_file.exists():
        return None

    print(f"📂 Loading saved predictions from: {predictions_file}")
    with open(predictions_file, 'rb') as f:
        predictions_data = pickle.load(f)

    return predictions_data


def regenerate_predictions(version, classifier_type, dataset_variant, model_dir):
    """Regenerate predictions from saved model and test data."""
    print(f"🔄 Regenerating predictions from saved model...")

    # Try loading from test_set_data.pkl first (random sampling mode)
    test_data_file = model_dir / 'test_evaluation' / 'test_set_data.pkl'

    if test_data_file.exists():
        print(f"📂 Loading test data from: {test_data_file}")
        with open(test_data_file, 'rb') as f:
            test_data = pickle.load(f)

        X_test = test_data['X_test']
        y_test = test_data['y_test']
        test_files = test_data.get('test_files', None)
        classifier_type = test_data['classifier_type']

    elif dataset_variant:
        # Dataset variant mode - load test data from CSV
        print(f"📂 Loading test data from dataset variant: {dataset_variant}")
        from ml.config import load_dataset_variant_info
        import pandas as pd
        import cv2

        # Load dataset variant info
        dataset_info = load_dataset_variant_info(dataset_variant)
        test_csv = dataset_info['dataset_dir'] / 'test.csv'

        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")

        # Read test CSV
        df_test = pd.read_csv(test_csv, encoding='utf-8')

        # Get data directory from dataset config
        dataset_config_path = model_dir / 'dataset_config.json'
        if dataset_config_path.exists():
            import json
            with open(dataset_config_path, 'r', encoding='utf-8') as f:
                dataset_config = json.load(f)
            data_dir = dataset_config.get('data_dir')
        else:
            data_dir = dataset_info['config'].get('data_dir')

        if not data_dir:
            raise ValueError("Cannot determine data directory")

        print(f"📂 Data directory: {data_dir}")
        print(f"📊 Test samples in CSV: {len(df_test)}")

        # Load test images
        test_files = []
        test_labels = []
        test_images = []

        for _, row in df_test.iterrows():
            filename = row['filename']
            label = int(row['has_porosity'])
            file_path = Path(data_dir) / filename

            if file_path.exists():
                # Load image
                img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    test_images.append(img)
                    test_files.append(filename)  # Store just filename, not full path
                    test_labels.append(label)

        if not test_images:
            raise ValueError("No test images loaded")

        # Convert to arrays and normalize
        X_test = np.array(test_images).astype('float32') / 255.0

        # Add channel dimension for CWT images
        if classifier_type == 'cwt_image':
            if len(X_test.shape) == 3:
                X_test = X_test[..., np.newaxis]

        y_test = np.array(test_labels)

        print(f"✅ Loaded {len(test_images)} test images")

    else:
        raise FileNotFoundError(
            f"Test data file not found: {test_data_file}\n"
            f"For dataset variant mode, please specify --dataset_variant"
        )

    # Load model
    model_file = model_dir / f'best_model_{version}.h5'
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"🤖 Loading model from: {model_file}")
    from tensorflow import keras
    model = keras.models.load_model(model_file, compile=False)

    # Load evaluation results to get best threshold
    eval_file = model_dir / 'test_evaluation' / f'comprehensive_evaluation_{version}.json'
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    import json
    with open(eval_file, 'r') as f:
        eval_results = json.load(f)

    best_threshold = eval_results['best_threshold']
    print(f"✅ Using threshold: {best_threshold:.3f}")

    # Get predictions
    print(f"🔮 Generating predictions...")
    if classifier_type == 'cwt_image':
        y_proba = model.predict(X_test, verbose=0)
    else:  # pd_signal
        pd1_test, pd2_test = X_test
        y_proba = model.predict([pd1_test, pd2_test], verbose=0)

    y_proba_flat = y_proba.flatten()
    y_pred = (y_proba_flat >= best_threshold).astype(int)

    # Compile predictions data
    predictions_data = {
        'y_pred': y_pred,
        'y_proba': y_proba_flat,
        'y_true': y_test,
        'best_threshold': best_threshold,
        'test_files': test_files,
        'classifier_type': classifier_type
    }

    return predictions_data


def generate_track_predictions_viz(test_files, y_true, y_pred, output_dir, version, exclude_final_window=False, use_time_labels=False):
    """
    Generate track-level prediction visualizations.

    Creates one figure per track showing a single row of colored boxes:
    - Blue (solid): True Negative (correctly predicted no porosity)
    - Red (solid): True Positive (correctly predicted porosity)
    - Blue (hatched): False Negative (missed porosity)
    - Red (hatched): False Positive (incorrectly predicted porosity)
    - White: Four skipped windows before first data window (no ticks/labels)
    - White: Four skipped windows after last data window (no ticks/labels)
    - White: Final data window if exclude_final_window=True

    X-axis ticks and labels only appear under data windows, starting from 0.

    Args:
        test_files: List of test file paths
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Output directory path
        version: Version string
        exclude_final_window: If True, white out and exclude final window from each track
        use_time_labels: If True, show time (ms) instead of window index on x-axis

    Returns:
        dict: Summary of generated visualizations
    """
    print(f"\n📊 Generating track-level prediction visualizations...")

    # Create output directory for track visualizations
    track_viz_dir = Path(output_dir) / 'test_evaluation' / 'track_predictions'
    track_viz_dir.mkdir(exist_ok=True, parents=True)

    # Extract track IDs from filenames (format: TRACKID_layerinfo_timewindow.png)
    # Example: 0105_01_0.2-1.2ms.png -> track_id = "0105_01"
    track_data = defaultdict(lambda: {'files': [], 'true_labels': [], 'pred_labels': [], 'windows': []})

    for i, filepath in enumerate(test_files):
        filename = Path(filepath).name

        # Extract track ID (first two underscore-separated parts)
        parts = filename.split('_')
        if len(parts) >= 2:
            track_id = f"{parts[0]}_{parts[1]}"

            # Extract time window info for sorting
            window_match = re.search(r'(\d+\.\d+)-(\d+\.\d+)ms', filename)
            if window_match:
                window_start = float(window_match.group(1))
            else:
                window_start = i  # Fallback to index

            track_data[track_id]['files'].append(filename)
            track_data[track_id]['true_labels'].append(y_true[i])
            track_data[track_id]['pred_labels'].append(y_pred[i])
            track_data[track_id]['windows'].append(window_start)

    print(f"Found {len(track_data)} unique tracks")

    # Generate visualization for each track
    figures_generated = []
    for track_id, data in sorted(track_data.items()):
        try:
            # Sort by window start time
            sorted_indices = np.argsort(data['windows'])
            true_labels = np.array(data['true_labels'])[sorted_indices]
            pred_labels = np.array(data['pred_labels'])[sorted_indices]
            windows = np.array(data['windows'])[sorted_indices]
            filenames = np.array(data['files'])[sorted_indices]

            n_windows = len(true_labels)
            n_skip_windows_start = 4  # Four white windows before first data window
            n_skip_windows_end = 4    # Four white windows after last data window

            # Create figure with fixed width of 6 inches
            fig, ax = plt.subplots(1, 1, figsize=(6, 0.5))

            # Add four skipped windows at the start (white cells at positions 0-3, no text, no ticks)
            for i in range(n_skip_windows_start):
                ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                          facecolor='white',
                                          edgecolor='black',
                                          linewidth=0.5))

            # Determine which windows to include in accuracy calculation
            if exclude_final_window and n_windows > 0:
                # Exclude final window from accuracy calculation
                n_windows_for_accuracy = n_windows - 1
                true_labels_for_accuracy = true_labels[:-1]
                pred_labels_for_accuracy = pred_labels[:-1]
            else:
                n_windows_for_accuracy = n_windows
                true_labels_for_accuracy = true_labels
                pred_labels_for_accuracy = pred_labels

            # Plot prediction results for each window (starting after skip windows)
            for i in range(n_windows):
                # Check if this is the final window and should be excluded
                if exclude_final_window and i == n_windows - 1:
                    # White out final window (no text)
                    ax.add_patch(plt.Rectangle((i + n_skip_windows_start, 0), 1, 1,
                                              facecolor='white',
                                              edgecolor='black',
                                              linewidth=0.5))
                else:
                    y_t = true_labels[i]
                    y_p = pred_labels[i]

                    # Determine color and hatching based on true vs predicted
                    if y_t == 0 and y_p == 0:
                        # True Negative - blue solid
                        color = COLOR_NO_POROSITY
                        hatch = None
                    elif y_t == 1 and y_p == 1:
                        # True Positive - red solid
                        color = COLOR_POROSITY
                        hatch = None
                    elif y_t == 1 and y_p == 0:
                        # False Negative - blue hatched (missed porosity)
                        color = COLOR_NO_POROSITY
                        hatch = '///'
                    else:  # y_t == 0 and y_p == 1
                        # False Positive - red hatched (false alarm)
                        color = COLOR_POROSITY
                        hatch = '///'

                    # Draw rectangle at position i+n_skip_windows_start
                    rect = plt.Rectangle((i + n_skip_windows_start, 0), 1, 1,
                                        facecolor=color,
                                        edgecolor='black',
                                        linewidth=0.5,
                                        hatch=hatch)
                    ax.add_patch(rect)

            # Add four skipped windows at the end (white cells, no text, no ticks)
            for i in range(n_skip_windows_end):
                ax.add_patch(plt.Rectangle((n_skip_windows_start + n_windows + i, 0), 1, 1,
                                          facecolor='white',
                                          edgecolor='black',
                                          linewidth=0.5))

            # Configure axes
            ax.set_xlim(0, n_skip_windows_start + n_windows + n_skip_windows_end)
            ax.set_ylim(0, 1)
            ax.set_yticks([])

            # X-axis configuration - ticks and labels only for data windows (not skip windows)
            # Exclude final window tick/label if exclude_final_window is True
            n_windows_for_ticks = n_windows - 1 if exclude_final_window else n_windows

            if use_time_labels:
                # Extract time windows and calculate center times
                time_labels = []
                for i in range(n_windows_for_ticks):
                    filename = filenames[i]
                    window_match = re.search(r'(\d+\.\d+)-(\d+\.\d+)ms', filename)
                    if window_match:
                        start_time = float(window_match.group(1))
                        end_time = float(window_match.group(2))
                        center_time = (start_time + end_time) / 2.0
                        time_labels.append(f'{center_time:.1f}')
                    else:
                        time_labels.append(str(i))

                ax.set_xlabel('Time (ms)', fontsize=10)
                ax.set_xticks(np.arange(n_skip_windows_start + 0.5, n_skip_windows_start + n_windows_for_ticks, 1))
                ax.set_xticklabels(time_labels, fontsize=8)

                # Hide some tick labels if there are many windows
                if n_windows_for_ticks > 25:
                    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
                        if idx % 3 != 0:
                            label.set_visible(False)
                            
                elif n_windows_for_ticks > 15:
                    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
                        if idx % 2 != 0:
                            label.set_visible(False)
                
                
                
                
            else:
                # Use window index starting from 0
                ax.set_xlabel('Time Window Index', fontsize=10)
                ax.set_xticks(np.arange(n_skip_windows_start + 0.5, n_skip_windows_start + n_windows_for_ticks, 1))
                ax.set_xticklabels(list(range(0, n_windows_for_ticks)), fontsize=8)

                # Hide every other tick label if there are many windows
                if n_windows_for_ticks > 20:
                    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
                        if idx % 2 == 1:
                            label.set_visible(False)

            # Add title above plot area with accuracy info
            if n_windows_for_accuracy > 0:
                accuracy = np.mean(true_labels_for_accuracy == pred_labels_for_accuracy)
                correct = np.sum(true_labels_for_accuracy == pred_labels_for_accuracy)
                total = len(true_labels_for_accuracy)
                title_text = f'Track: {track_id}     Accuracy: {accuracy:.1%} ({correct}/{total})'
            else:
                title_text = f'Track: {track_id}     No windows for accuracy'

            ax.set_title(title_text, fontsize=10, fontweight='bold', pad=10)

            # Create legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLOR_NO_POROSITY, edgecolor='black', label='True Negative'),
                Patch(facecolor=COLOR_POROSITY, edgecolor='black', label='True Positive'),
                Patch(facecolor=COLOR_NO_POROSITY, edgecolor='black', hatch='///', label='False Negative'),
                Patch(facecolor=COLOR_POROSITY, edgecolor='black', hatch='///', label='False Positive'),
                Patch(facecolor='white', edgecolor='black', label='Skipped Window')
            ]
            
            # Shrink plot to make space for legend below
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.5,
                             box.width, box.height * 0.5])

            # Place legend well below the plot
            ax.legend(handles=legend_elements, loc='upper center',
                     bbox_to_anchor=(0.5, -2), ncol=5, fontsize=8, frameon=True)

            # Save figure (bbox_inches='tight' handles spacing better than tight_layout)
            output_file = track_viz_dir / f'track_{track_id}_predictions.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            figures_generated.append(str(output_file))

        except Exception as e:
            print(f"Warning: Could not generate visualization for track {track_id}: {e}")
            continue

    print(f"✅ Generated {len(figures_generated)} track prediction visualizations")
    print(f"📁 Output directory: {track_viz_dir}")

    return {
        'total_tracks': len(track_data),
        'figures_generated': len(figures_generated),
        'output_directory': str(track_viz_dir)
    }


def generate_confusion_matrix(y_true, y_pred, output_dir, version, threshold, test_files=None, exclude_final_window=False, class_labels=None, subdir='test_evaluation'):
    """
    Generate a normalized confusion matrix with percentage values.

    Formatted for 1/2 A4 width at 300 DPI with 9pt text.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Output directory path
        version: Version string
        threshold: Classification threshold used
        test_files: List of test file paths (for identifying final windows per track)
        exclude_final_window: If True, exclude final window of each track
        class_labels: List of class label strings (default: ['No Porosity', 'Porosity'])
        subdir: Subdirectory name within output_dir (default: 'test_evaluation')

    Returns:
        str: Path to saved confusion matrix file
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Default class labels if not provided
    if class_labels is None:
        class_labels = ['No Porosity', 'Porosity']

    print(f"\n📊 Generating confusion matrix...")

    # Filter out final windows if requested
    if exclude_final_window and test_files is not None:
        # Group files by track ID to identify final windows
        from collections import defaultdict
        track_indices = defaultdict(list)

        for i, filepath in enumerate(test_files):
            filename = Path(filepath).name
            parts = filename.split('_')
            if len(parts) >= 2:
                track_id = f"{parts[0]}_{parts[1]}"

                # Extract time window for sorting
                window_match = re.search(r'(\d+\.\d+)-(\d+\.\d+)ms', filename)
                window_start = float(window_match.group(1)) if window_match else i

                track_indices[track_id].append((i, window_start))

        # Find indices of final windows for each track
        final_window_indices = set()
        for track_id, indices_and_times in track_indices.items():
            # Sort by time and get index of last window
            sorted_items = sorted(indices_and_times, key=lambda x: x[1])
            if sorted_items:
                final_idx = sorted_items[-1][0]
                final_window_indices.add(final_idx)

        # Create mask to exclude final windows
        mask = np.array([i not in final_window_indices for i in range(len(y_true))])
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        print(f"   Excluded {len(final_window_indices)} final windows from confusion matrix")



    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize to percentages (row-wise)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # A4 width is 210mm, half is 105mm = 4.13 inches at 300 DPI
    fig_width = 4.13
    fig_height = 3.5  # Maintain reasonable aspect ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap without annotations first
    sns.heatmap(cm_normalized, annot=False, fmt='', cmap='Blues',
                cbar_kws={'label': 'Percentage (%)'},
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax, square=True, vmin=0, vmax=100)

    # Manually add annotations with custom font sizes and adaptive text color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Determine text color based on background intensity (>50% = white, <=50% = black)
            percentage = cm_normalized[i, j]
            text_color = 'white' if percentage > 50 else 'black'

            # Percentage text (9pt, regular weight)
            ax.text(j + 0.5, i + 0.42, f'{percentage:.1f}%',
                   ha='center', va='center', fontsize=9, color=text_color)
            # Small count text (7pt, closer to percentage)
            ax.text(j + 0.5, i + 0.60, f'n={cm[i, j]}',
                   ha='center', va='center', fontsize=7, color=text_color)

    # Set labels with 9pt font
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)
    ax.set_title('Confusion Matrix', fontsize=10, fontweight='bold')

    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=9)

    # Set colorbar label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Percentage (%)', fontsize=9)

    plt.tight_layout()

    # Save figure at 300 DPI
    output_file = Path(output_dir) / subdir / f'confusion_matrix_{version}.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure subdirectory exists
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrix saved to: {output_file}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate track-level prediction visualizations from saved model outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use saved predictions (fast)
  python ml/visualize_track_predictions.py --version v001 --classifier_type cwt_image

  # With dataset variant
  python ml/visualize_track_predictions.py --version v115 --dataset_variant AlSi10Mg_CW_L1_powder_porosity_with-test_auto-split

  # Force regeneration from model
  python ml/visualize_track_predictions.py --version v001 --force-regenerate
        """
    )

    parser.add_argument('--version', type=str, required=True,
                       help='Model version (e.g., v001, v115, or just 1, 115)')
    parser.add_argument('--classifier_type', type=str, choices=['cwt_image', 'pd_signal'],
                       default='cwt_image', help='Type of classifier (default: cwt_image)')
    parser.add_argument('--dataset_variant', type=str, default=None,
                       help='Dataset variant name (optional, used for finding model directory)')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regeneration of predictions from model even if saved predictions exist')
    parser.add_argument('--exclude-final-window', action='store_true',
                       help='Exclude final window of each track from visualizations and confusion matrix')
    parser.add_argument('--use-time-labels', action='store_true',
                       help='Use time (ms) labels instead of window index on x-axis')

    args = parser.parse_args()

    # Format version
    version = format_version(args.version)

    print(f"{'='*70}")
    print(f"📊 Track Predictions Visualizer")
    print(f"{'='*70}")
    print(f"Version: {version}")
    print(f"Classifier: {args.classifier_type}")
    if args.dataset_variant:
        print(f"Dataset: {args.dataset_variant}")
    print(f"{'='*70}\n")

    # Get model directory (base directory, then add version)
    base_model_dir = get_model_dir(args.classifier_type, args.dataset_variant)
    model_dir = base_model_dir / version

    if not model_dir.exists():
        print(f"❌ Error: Model version directory not found: {model_dir}")
        print(f"   Base directory: {base_model_dir}")
        print(f"   Looking for version: {version}")
        sys.exit(1)

    print(f"📂 Model directory: {model_dir}\n")

    # Auto-detect dataset variant if not specified
    dataset_variant = args.dataset_variant
    if not dataset_variant:
        dataset_config_path = model_dir / 'dataset_config.json'
        if dataset_config_path.exists():
            import json
            try:
                with open(dataset_config_path, 'r', encoding='utf-8') as f:
                    dataset_config = json.load(f)
                    dataset_variant = dataset_config.get('dataset_name')
                    if dataset_variant:
                        print(f"🔍 Auto-detected dataset variant: {dataset_variant}")
            except Exception as e:
                pass  # Failed to auto-detect, will try pkl file

    # Load or regenerate predictions
    predictions_data = None

    if not args.force_regenerate:
        predictions_data = load_saved_predictions(version, model_dir)

    if predictions_data is None:
        print(f"⚠️  Saved predictions not found or regeneration forced")
        predictions_data = regenerate_predictions(
            version, args.classifier_type, dataset_variant, model_dir
        )

    # Extract data
    y_true = predictions_data['y_true']
    y_pred = predictions_data['y_pred']
    test_files = predictions_data['test_files']
    best_threshold = predictions_data['best_threshold']

    if test_files is None:
        print(f"❌ Error: No test file names available in predictions data")
        sys.exit(1)

    print(f"\n📊 Data Summary:")
    print(f"   Test samples: {len(y_true)}")
    print(f"   Threshold: {best_threshold:.3f}")
    print(f"   Overall accuracy: {np.mean(y_true == y_pred):.1%}")

    # Generate track predictions visualizations
    track_results = generate_track_predictions_viz(
        test_files, y_true, y_pred, model_dir, version, args.exclude_final_window, args.use_time_labels
    )

    # Generate confusion matrix
    cm_file = generate_confusion_matrix(
        y_true, y_pred, model_dir, version, best_threshold, test_files, args.exclude_final_window
    )

    print(f"\n{'='*70}")
    print(f"✅ Visualization Complete!")
    print(f"{'='*70}")
    print(f"📊 Tracks visualized: {track_results['figures_generated']}/{track_results['total_tracks']}")
    print(f"📁 Track predictions: {track_results['output_directory']}")
    print(f"📊 Confusion matrix: {cm_file}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
