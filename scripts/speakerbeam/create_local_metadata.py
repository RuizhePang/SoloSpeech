#!/usr/bin/env python3

# This code is adapted from SpeakerBeam/Asteroid create_local_metadata.py.
# SpeakerBeam source:
# https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/create_local_metadata.py

import argparse
import csv
import os
from glob import glob


def normalize_librimix_path(path, librimix_dir):
    """Rewrite stale absolute Libri2Mix paths to the current Libri2Mix root."""
    if not path:
        return path

    parts = path.split(os.sep)
    for idx, part in enumerate(parts):
        if part.startswith("wav") and idx + 2 < len(parts):
            return os.path.join(librimix_dir, *parts[idx:])

    marker = f"{os.sep}Libri2Mix{os.sep}"
    if marker in path:
        return os.path.join(librimix_dir, path.split(marker, 1)[1])

    return path


def copy_metadata_with_current_paths(src_csv, dst_csv, librimix_dir):
    with open(src_csv, newline="") as src, open(dst_csv, "w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            for field in reader.fieldnames:
                if field.endswith("_path"):
                    row[field] = normalize_librimix_path(row[field], librimix_dir)
            writer.writerow(row)


def create_local_metadata(librimix_dir, out_dir):
    librimix_dir = os.path.abspath(librimix_dir)
    md_dirs = [
        path
        for path in glob(os.path.join(librimix_dir, "*/*/*"))
        if path.endswith("metadata")
    ]

    for md_dir in md_dirs:
        md_files = [path for path in os.listdir(md_dir) if path.startswith("mix")]
        for md_file in md_files:
            subset = md_file.split("_")[1]
            rel_parts = os.path.relpath(md_dir, librimix_dir).split(os.sep)
            if rel_parts[-1] == "metadata":
                rel_parts = rel_parts[:-1]
            local_path = os.path.join(out_dir, *rel_parts, subset)
            os.makedirs(local_path, exist_ok=True)
            copy_metadata_with_current_paths(
                os.path.join(md_dir, md_file),
                os.path.join(local_path, md_file),
                librimix_dir,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librimix_dir",
        type=str,
        required=True,
        help="Path to LibriMix/Libri2Mix directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where SpeakerBeam-style data csv files are written",
    )
    args = parser.parse_args()

    create_local_metadata(args.librimix_dir, args.out_dir)


if __name__ == "__main__":
    main()
