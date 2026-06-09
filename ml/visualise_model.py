#!/usr/bin/env python3
"""
Visualise a saved CWT classifier model by version number.

Usage:
    python ml/visualise_model.py v115
    python ml/visualise_model.py v115 --fold 1   # use fold-specific model

Outputs:
    ml/outputs/cwt/<version>/model_architecture_<version>.png
"""
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# ── Path config ───────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path('ml/outputs/cwt')

def find_model(version: str, fold: int | None) -> Path:
    vdir = OUTPUTS_DIR / version
    if not vdir.exists():
        sys.exit(f'ERROR: No output directory for {version} at {vdir}')

    if fold is not None:
        path = vdir / 'models' / f'best_model_fold_{fold}.h5'
    else:
        path = vdir / f'best_model_{version}.h5'

    if not path.exists():
        sys.exit(f'ERROR: Model file not found: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description='Visualise a saved Keras model')
    parser.add_argument('version', help='Version string, e.g. v115')
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number (default: use best_model_<version>.h5)')
    args = parser.parse_args()

    model_path = find_model(args.version, args.fold)
    print(f'Loading: {model_path}')

    model = tf.keras.models.load_model(str(model_path))

    # ── Text summary ─────────────────────────────────────────────────────────
    print()
    model.summary()

    # ── Graphical plot ────────────────────────────────────────────────────────
    out_path = OUTPUTS_DIR / args.version / f'model_architecture_{args.version}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_model(
        model,
        to_file=str(out_path),
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='TB',       # top-to-bottom; change to 'LR' for left-to-right
        dpi=150,
        expand_nested=False,
    )
    print(f'\nSaved architecture plot: {out_path}')


if __name__ == '__main__':
    main()
