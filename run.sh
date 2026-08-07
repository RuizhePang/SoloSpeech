#!/bin/bash

set -euo pipefail

source ./tools/path.sh

cmd="run.pl"
data_dir="$PWD/data/train"
scheduler_arguments=()

if [[ "${HOSTNAME:-}" == asp2a* ]]; then
    cmd="pbs.pl"
    data_dir="$HOME/scratch/train"

    project="personal-n2602009"; name="SoloSpeech"; qos="normal"; n_nodes=1; n_cpus=16; n_gpus=1; time=48:00:00; mem=64G

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

config=SoloSpeech
save_dir="experiments/${config}"

log_dir="$save_dir/logs"

export PYTHONPATH="$PWD/solospeech/stable_audio_vae:${PYTHONPATH:-}"

explog="$log_dir/train.vae.log"
"$cmd" "${scheduler_arguments[@]}" "$explog" \
    "$PYTHON_BIN" scripts/train/vae.py \
    --config-name="${config}" \
    --save_dir="${save_dir}" \
    --data_dir="${data_dir}"
