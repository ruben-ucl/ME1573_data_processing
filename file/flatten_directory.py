"""
Move all files from immediate subfolders up to the parent directory,
then delete the now-empty subfolders.

Usage:
    python file/flatten_directory.py <directory> [--dry_run]
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path,
                        help='Directory whose subfolders will be flattened')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print actions without making any changes')
    return parser.parse_args()


def flatten(directory: Path, dry_run: bool):
    subfolders = [p for p in directory.iterdir() if p.is_dir()]

    if not subfolders:
        print('No subfolders found.')
        return

    conflicts = []
    moves = []

    for folder in subfolders:
        for src in folder.iterdir():
            if src.is_file():
                dst = directory / src.name
                if dst.exists():
                    conflicts.append((src, dst))
                else:
                    moves.append((src, dst, folder))

    if conflicts:
        print(f'Aborting: {len(conflicts)} filename conflict(s) would overwrite existing files:')
        for src, dst in conflicts:
            print(f'  {src}  ->  {dst}  (already exists)')
        sys.exit(1)

    for src, dst, _ in moves:
        print(f'  {"[dry]" if dry_run else ""} move  {src}  ->  {dst}')
        if not dry_run:
            shutil.move(str(src), dst)

    empty_folders = {folder for _, _, folder in moves}
    for folder in subfolders:
        if folder not in empty_folders:
            empty_folders.add(folder)

    for folder in sorted(empty_folders):
        remaining = list(folder.iterdir())
        if remaining:
            print(f'  Skipping delete of {folder} ({len(remaining)} item(s) remain)')
            continue
        print(f'  {"[dry]" if dry_run else ""} rmdir {folder}')
        if not dry_run:
            folder.rmdir()

    if dry_run:
        print(f'\nDry run: {len(moves)} file(s) would be moved, no changes made.')
    else:
        print(f'\nDone: {len(moves)} file(s) moved.')


def main():
    args = parse_args()
    directory = args.directory.resolve()

    if not directory.is_dir():
        print(f'Error: {directory} is not a directory.')
        sys.exit(1)

    print(f'Flattening: {directory}{"  [dry run]" if args.dry_run else ""}\n')
    flatten(directory, args.dry_run)


if __name__ == '__main__':
    main()
