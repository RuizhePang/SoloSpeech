#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <librimix_dir> <out_dir> <python_bin> [sample_rate] [mix_mode] [task]" >&2
    echo "Example: $0 data/train/Libri2Mix/LibriMixData/Libri2Mix data/train/Libri2Mix/SpeakerBeamData python 16k min mix_both" >&2
    exit 1
fi

librimix_dir=$1
out_dir=$2
python_bin=$3
sample_rate=${4:-"16k"}
mix_mode=${5:-"min"}
task=${6:-"mix_both"}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
speakerbeam_scripts="${script_dir}/speakerbeam"
data_root="${out_dir}/wav${sample_rate}/${mix_mode}"

download_file() {
    local url=$1
    local out_path=$2

    mkdir -p "$(dirname "${out_path}")"
    if [[ -s "${out_path}" ]]; then
        echo "${out_path} already exists. Skipping download."
        return
    fi

    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 5 --retry-delay 5 -o "${out_path}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${out_path}" "${url}"
    else
        echo "Neither curl nor wget is available for downloading ${url}" >&2
        exit 1
    fi
}

create_eval_map() {
    local subset=$1
    local url="https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/${subset}/map_mixture2enrollment"
    local map_path="${data_root}/${subset}/map_mixture2enrollment"

    download_file "${url}" "${map_path}"
}

create_enrollment_for_subset() {
    local subset=$1
    local mix_csv="${data_root}/${subset}/mixture_${subset}_${task}.csv"
    local out_csv="${data_root}/${subset}/mixture2enrollment.csv"

    if [[ ! -f "${mix_csv}" ]]; then
        echo "Missing ${mix_csv}; did stage0 generate wav${sample_rate}/${mix_mode}/${subset}/${task}?" >&2
        exit 1
    fi

    case "${subset}" in
        train-100|train-360)
            "${python_bin}" "${speakerbeam_scripts}/create_enrollment_csv_all.py" \
                "${mix_csv}" \
                "${out_csv}"
            ;;
        dev|test)
            create_eval_map "${subset}"
            "${python_bin}" "${speakerbeam_scripts}/create_enrollment_csv_fixed.py" \
                "${mix_csv}" \
                "${data_root}/${subset}/map_mixture2enrollment" \
                "${out_csv}"
            ;;
        *)
            echo "Unsupported Libri2Mix subset: ${subset}" >&2
            exit 1
            ;;
    esac
}

if [[ ! -d "${librimix_dir}/wav${sample_rate}/${mix_mode}" ]]; then
    echo "Missing Libri2Mix directory: ${librimix_dir}/wav${sample_rate}/${mix_mode}" >&2
    exit 1
fi

mkdir -p "${out_dir}"

"${python_bin}" "${speakerbeam_scripts}/create_local_metadata.py" \
    --librimix_dir "${librimix_dir}" \
    --out_dir "${out_dir}"

create_enrollment_for_subset train-100
create_enrollment_for_subset train-360
create_enrollment_for_subset dev
create_enrollment_for_subset test

echo "SpeakerBeam-style Libri2Mix data prepared in ${data_root}"
