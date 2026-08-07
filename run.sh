#!/bin/bash

set -euo pipefail

source ./tools/path.sh

cmd="run.pl"
local_dir="$PWD/train"
scheduler_arguments=()

if [[ "${HOSTNAME:-}" == asp2a* ]]; then
    cmd="pbs.pl"
    local_dir="$HOME/scratch/train"

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

"$cmd" \
    "${scheduler_arguments[@]}" \
    "./test.log" \
    "$PYTHON_BIN" test.py
