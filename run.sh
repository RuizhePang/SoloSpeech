#!/bin/bash

set -euo pipefail

source ./tools/path.sh

cmd="run.pl"
data_dir="$PWD/data/train"
scheduler_arguments=()

if [[ "${HOSTNAME:-}" == asp2a* ]]; then
    cmd="pbs.pl"
    data_dir="$HOME/scratch/train"

    project="personal-n2602009"; name="SoloSpeech"; qos="normal"; n_nodes=1; n_cpus=10; n_gpus=1; time=12:00:00; mem=64G

    scheduler_arguments=(
        -P ${project}
        -N ${name}
        -q ${qos}
        -l select=${n_nodes}:ncpus=${n_cpus}:ngpus=${n_gpus}:mem=${mem}
        -l walltime=${time}
        -o /dev/null
        -e /dev/null
    )
fi

stage=0
stop_stage=100

config=SoloSpeech

save_dir="experiments/${config}"
log_dir="$save_dir/logs"

. ./tools/parse_option.sh

mkdir -p "$log_dir"

if [[ $stage -le 0 && $stop_stage -ge 0 ]]; then
    echo "Stage 0: downloading/generating Libri2Mix into $data_dir/Libri2Mix"
    bash scripts/download_librimix.sh \
        "$data_dir/Libri2Mix" \
        "$PYTHON_BIN" \
        "16k" \
        "min" \
        "mix_clean mix_both"
fi

if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
    echo "Stage 1: preparing SpeakerBeam-style TSE data"
    bash scripts/prepare_speakerbeam_librimix.sh \
        "$data_dir/Libri2Mix/LibriMixData/Libri2Mix" \
        "$data_dir/Libri2Mix/SpeakerBeamData" \
        "$PYTHON_BIN" \
        "16k" \
        "min" \
        "mix_both"
fi

if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
    export PYTHONPATH="$PWD/solospeech/stable_audio_vae:${PYTHONPATH:-}"
    explog="$log_dir/train.vae.log"
    if [[ -f "$explog" ]]; then
        echo "Log file $explog already exists. Please remove it before running the script."
        exit 1
    fi
    echo "Stage 2: running VAE training with config: $config
    Logging to: $explog"
    "$cmd" "${scheduler_arguments[@]}" "$explog" \
        "$PYTHON_BIN" scripts/train/vae.py \
        --config-name="${config}" \
        save_dir="${save_dir}/compressor" \
        data_dir="${data_dir}"
fi
