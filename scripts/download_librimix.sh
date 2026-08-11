#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <librimix_root> <python_bin> [freqs] [modes] [types] [sample_ratio]" >&2
    echo "Example: $0 data/train/Libri2Mix python '16k' 'min' 'mix_clean mix_both' '1/1000'" >&2
    exit 1
fi

librimix_root=$1
python_bin=$2
freqs=${3:-"16k"}
modes=${4:-"min"}
types=${5:-"mix_clean mix_both"}
sample_ratio=${6:-"1"}

librimix_repo_url=${LIBRIMIX_REPO_URL:-"https://github.com/JorisCos/LibriMix.git"}
librimix_repo_dir=${LIBRIMIX_REPO_DIR:-"${librimix_root}/LibriMix"}
librimix_storage_dir=${LIBRIMIX_STORAGE_DIR:-"${librimix_root}/LibriMixData"}
metadata_dir="${librimix_root}/metadata/Libri2Mix.sampled"

librispeech_dir="${librimix_storage_dir}/LibriSpeech"
wham_dir="${librimix_storage_dir}/wham_noise"

mkdir -p "${librimix_root}" "${librimix_storage_dir}"

if ! command -v sox >/dev/null 2>&1; then
    echo "SoX is required by the official LibriMix noise augmentation script." >&2
    echo "Install it before running stage0, e.g. conda install -c conda-forge sox." >&2
    exit 1
fi

"${python_bin}" - <<'PY'
import importlib.util
import sys

missing = [
    module
    for module in ("soundfile", "pandas", "numpy", "scipy", "tqdm", "pysndfx", "pyloudnorm")
    if importlib.util.find_spec(module) is None
]
if missing:
    print(
        "Missing Python packages required by LibriMix generation: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print("Install the updated requirements.txt before running stage0.", file=sys.stderr)
    sys.exit(1)
PY

download() {
    local url=$1
    local out_dir=$2
    local out_path="${out_dir}/$(basename "${url}")"

    if command -v wget >/dev/null 2>&1; then
        wget -c --tries=0 --read-timeout=20 "${url}" -P "${out_dir}"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL --retry 10 --retry-delay 10 -C - -o "${out_path}" "${url}"
    else
        echo "Neither wget nor curl is available for downloading ${url}" >&2
        exit 1
    fi
}

ensure_librimix_repo() {
    if [[ -d "${librimix_repo_dir}/.git" ]]; then
        git -C "${librimix_repo_dir}" fetch --depth 1 origin master
        git -C "${librimix_repo_dir}" checkout -q FETCH_HEAD
    else
        git clone --depth 1 "${librimix_repo_url}" "${librimix_repo_dir}"
    fi
}

prepare_sampled_metadata() {
    "${python_bin}" - \
        "${librimix_repo_dir}/metadata/Libri2Mix" \
        "${metadata_dir}" \
        "${sample_ratio}" <<'PY'
import csv
import math
import random
import shutil
import sys
from pathlib import Path

src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])
ratio_text = sys.argv[3]

def parse_ratio(text):
    text = text.strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)

ratio = parse_ratio(ratio_text)
if ratio <= 0 or ratio > 1:
    raise SystemExit(f"sample_ratio must be in (0, 1], got: {ratio_text}")

dst_dir.mkdir(parents=True, exist_ok=True)
for old_file in dst_dir.glob("*.csv"):
    old_file.unlink()

for src_file in sorted(src_dir.glob("*.csv")):
    dst_file = dst_dir / src_file.name
    if src_file.name.endswith("_info.csv"):
        shutil.copy2(src_file, dst_file)
        continue

    with src_file.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if ratio < 1:
        rng = random.Random(f"{src_file.name}:{ratio_text}")
        sample_size = max(1, math.ceil(len(rows) * ratio))
        rows = sorted(rng.sample(rows, sample_size))

    with dst_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"{src_file.name}: {len(rows)} rows")
PY
}

download_librispeech_split() {
    local split=$1
    local archive="${split}.tar.gz"

    if [[ -d "${librispeech_dir}/${split}" ]]; then
        echo "LibriSpeech/${split} already exists. Skipping."
        return
    fi

    echo "Downloading LibriSpeech/${split} into ${librimix_storage_dir}"
    download "http://www.openslr.org/resources/12/${archive}" "${librimix_storage_dir}"
    tar -xzf "${librimix_storage_dir}/${archive}" -C "${librimix_storage_dir}"
    rm -f "${librimix_storage_dir:?}/${archive}"
}

download_wham() {
    local archive="wham_noise.zip"

    if [[ -d "${wham_dir}" ]]; then
        echo "wham_noise already exists. Skipping."
        return
    fi

    echo "Downloading wham_noise into ${librimix_storage_dir}"
    download "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/${archive}" "${librimix_storage_dir}"
    unzip -qn "${librimix_storage_dir}/${archive}" -d "${librimix_storage_dir}"
    rm -f "${librimix_storage_dir:?}/${archive}"
}

ensure_librimix_repo
prepare_sampled_metadata

download_librispeech_split dev-clean &
download_librispeech_split test-clean &
download_librispeech_split train-clean-100 &
download_librispeech_split train-clean-360 &
download_wham &
wait

"${python_bin}" "${librimix_repo_dir}/scripts/augment_train_noise.py" --wham_dir "${wham_dir}"

read -r -a freq_args <<< "${freqs}"
read -r -a mode_args <<< "${modes}"
read -r -a type_args <<< "${types}"

echo "Generating Libri2Mix only:"
echo "  output: ${librimix_storage_dir}/Libri2Mix"
echo "  freqs: ${freq_args[*]}"
echo "  modes: ${mode_args[*]}"
echo "  types: ${type_args[*]}"
echo "  sample_ratio: ${sample_ratio}"

"${python_bin}" "${librimix_repo_dir}/scripts/create_librimix_from_metadata.py" \
    --librispeech_dir "${librispeech_dir}" \
    --wham_dir "${wham_dir}" \
    --metadata_dir "${metadata_dir}" \
    --librimix_outdir "${librimix_storage_dir}" \
    --n_src 2 \
    --freqs "${freq_args[@]}" \
    --modes "${mode_args[@]}" \
    --types "${type_args[@]}"

if [[ -d "${librimix_storage_dir}/Libri3Mix" ]]; then
    echo "Warning: ${librimix_storage_dir}/Libri3Mix already exists, but this script did not create or update it." >&2
fi
