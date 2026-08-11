#!/bin/bash

set -euo pipefail

source ./tools/path.sh

cmd="run.pl"
data_dir="$PWD/data/train"
scheduler_arguments=()

if [[ "${HOSTNAME:-}" == asp2a* ]]; then
    cmd="pbs.pl"
    data_dir="$HOME/scratch/train"

    qos="normal"
    project="personal-n2602009"; name="SoloSpeech"; n_nodes=1; n_cpus=10; n_gpus=1; time=15:00:00; mem=64G

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

stage=0 # Download and prepare Libri2Mix data
# stage=1 # Prepare SpeakerBeam-style TSE data
# stage=2 # Train VAE on Libri2Mix data
# stage=3 # Extract VAE embeddings for Libri2Mix data
# stage=4 # Train TSE extractor on Libri2Mix VAE embeddings
# stage=5 # Generate FastGECO corrector training data with extractor
# stage=6 # Train FastGECO corrector
# stage=7 # Evaluate compressor/extractor/corrector/system metrics

stop_stage=$stage

# config=pretrained
config=SoloSpeech
librimix_sample_ratio=0.0001
metrics=null
eval_config=default
test_dir=null
test_manifest=null

save_dir="experiments/${config}"
log_dir="$save_dir/logs"

. ./tools/parse_option.sh

evaluation_overrides=(evaluation="${eval_config}")
if [[ "$metrics" != "null" && -n "$metrics" ]]; then
    evaluation_overrides+=(evaluation.metrics="${metrics}")
fi
if [[ "$test_dir" != "null" && -n "$test_dir" ]]; then
    evaluation_overrides+=(evaluation.test_dir="${test_dir}")
fi
if [[ "$test_manifest" != "null" && -n "$test_manifest" ]]; then
    evaluation_overrides+=(evaluation.manifest="${test_manifest}")
fi

mkdir -p "$log_dir"

if [[ $stage -le 0 && $stop_stage -ge 0 ]]; then
    echo "Stage 0: downloading/generating Libri2Mix into $data_dir/Libri2Mix"
    bash scripts/download_librimix.sh \
        "$data_dir/Libri2Mix" \
        "$PYTHON_BIN" \
        "16k" \
        "min" \
        "mix_clean mix_both" \
        "$librimix_sample_ratio"
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
    explog="$log_dir/train.compressor.log"
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

if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
    export PYTHONPATH="$PWD:$PWD/solospeech/stable_audio_vae:${PYTHONPATH:-}"
    explog="$log_dir/data.compressor.log"
    if [[ -f "$explog" ]]; then
        echo "Log file $explog already exists. Please remove it before running the script."
        exit 1
    fi

    echo "Stage 3: encoding Libri2Mix wav files with VAE
    Logging to: $explog"
    "$cmd" "${scheduler_arguments[@]}" "$explog" \
        "$PYTHON_BIN" scripts/extract_vae.py \
        --config-name="${config}" \
        save_dir="${save_dir}/compressor" \
        data_dir="${data_dir}"
fi

if [[ $stage -le 4 && $stop_stage -ge 4 ]]; then
    export PYTHONPATH="$PWD:$PWD/solospeech/stable_audio_vae:${PYTHONPATH:-}"
    explog="$log_dir/train.extractor.log"
    if [[ -f "$explog" ]]; then
        echo "Log file $explog already exists. Please remove it before running the script."
        exit 1
    fi
    echo "Stage 4: running TSE extractor training with config: $config
    Logging to: $explog"
    "$cmd" "${scheduler_arguments[@]}" "$explog" \
        "$PYTHON_BIN" scripts/train/tse.py \
        --config-name="${config}" \
        save_dir="${save_dir}/extractor" \
        data_dir="${data_dir}"
fi

if [[ $stage -le 5 && $stop_stage -ge 5 ]]; then
    export PYTHONPATH="$PWD:$PWD/solospeech/stable_audio_vae:${PYTHONPATH:-}"
    explog="$log_dir/data.corrector.log"
    if [[ -f "$explog" ]]; then
        echo "Log file $explog already exists. Please remove it before running the script."
        exit 1
    fi
    echo "Stage 5: generating FastGECO corrector data with config: $config
    Logging to: $explog"
    "$cmd" "${scheduler_arguments[@]}" "$explog" \
        "$PYTHON_BIN" scripts/generate_corrector_data.py \
        --config-name="${config}" \
        save_dir="${save_dir}" \
        data_dir="${data_dir}"
fi

if [[ $stage -le 6 && $stop_stage -ge 6 ]]; then
    export PYTHONPATH="$PWD:$PWD/solospeech/stable_audio_vae:${PYTHONPATH:-}"
    explog="$log_dir/train.corrector.log"
    if [[ -f "$explog" ]]; then
        echo "Log file $explog already exists. Please remove it before running the script."
        exit 1
    fi
    echo "Stage 6: running FastGECO corrector training with config: $config
    Logging to: $explog"
    "$cmd" "${scheduler_arguments[@]}" "$explog" \
        "$PYTHON_BIN" scripts/train/corrector.py \
        --config-name="${config}" \
        save_dir="${save_dir}/corrector" \
        data_dir="${data_dir}"
fi

if [[ $stage -le 7 && $stop_stage -ge 7 ]]; then
    export PYTHONPATH="$PWD:$PWD/solospeech/stable_audio_vae:${PYTHONPATH:-}"
    explog="$log_dir/evaluation.log"
    if [[ -f "$explog" ]]; then
        echo "Log file $explog already exists. Please remove it before running the script."
        exit 1
    fi
    echo "Stage 7: running evaluation with config: $config eval_config: $eval_config test_dir: $test_dir manifest: $test_manifest metrics_override: $metrics
    Logging to: $explog"
    "$cmd" "${scheduler_arguments[@]}" "$explog" \
        "$PYTHON_BIN" scripts/evaluate.py \
        --config-name="${config}" \
        "${evaluation_overrides[@]}" \
        save_dir="${save_dir}" \
        data_dir="${data_dir}"
fi
