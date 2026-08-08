#!/usr/bin/env python3

# This code is adapted from SpeakerBeam/Asteroid create_local_metadata.py.
# SpeakerBeam source:
# https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/create_local_metadata.py

import argparse
import os
import shutil
from glob import glob


def create_local_metadata(librimix_dir, out_dir):
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
            shutil.copy(os.path.join(md_dir, md_file), local_path)


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
