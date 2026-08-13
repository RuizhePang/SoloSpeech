#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <out_dir> <python_bin> [array] [channel] [max_items]" >&2
    echo "Example: $0 data/train/CHiME6 python U01 CH1 0" >&2
    exit 1
fi

out_dir=$1
python_bin=$2
array=${3:-U01}
channel=${4:-CH1}
max_items=${5:-0}

if [[ -n "${CHIME6_MIRROR:-}" ]]; then
    chime6_mirrors=("${CHIME6_MIRROR}")
elif [[ -n "${CHIME6_MIRRORS:-}" ]]; then
    read -r -a chime6_mirrors <<<"${CHIME6_MIRRORS}"
else
    chime6_mirrors=(
        "https://openslr.trmal.net/resources/150"
        "https://openslr.elda.org/resources/150"
        "https://www.openslr.org/resources/150"
    )
fi
raw_dir="${out_dir}/raw"
processed_dir="${out_dir}/test"
manifest_path="${processed_dir}/manifest.csv"

mkdir -p "${raw_dir}" "${processed_dir}"

for tool in gzip tar; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "${tool} is required by CHiME-6 preparation." >&2
        exit 1
    fi
done

if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    echo "wget or curl is required for downloading CHiME-6." >&2
    exit 1
fi

"${python_bin}" - <<'PY'
import importlib.util
import sys

missing = [module for module in ("soundfile", "numpy") if importlib.util.find_spec(module) is None]
if missing:
    print("Missing Python packages required by CHiME-6 preparation: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

download() {
    local name=$1
    local kind=${2:-tar}
    local path="${raw_dir}/${name}"

    if [[ -f "${path}" ]] && is_valid_archive "${path}" "${kind}"; then
        echo "${path} already exists and is valid. Skipping download."
        return
    fi

    if [[ -f "${path}" ]]; then
        echo "${path} is not a valid ${kind} archive. Removing it before retrying." >&2
        rm -f "${path}"
    fi

    local mirror_url url
    for mirror_url in "${chime6_mirrors[@]}"; do
        url="${mirror_url%/}/${name}"
        echo "Downloading ${url}"
        rm -f "${path}"
        if command -v wget >/dev/null 2>&1; then
            wget --tries=5 --waitretry=5 --no-check-certificate -O "${path}" "${url}" || true
        else
            curl -fL --retry 5 --retry-delay 5 -o "${path}" "${url}" || true
        fi

        if [[ -f "${path}" ]] && is_valid_archive "${path}" "${kind}"; then
            return
        fi

        if [[ -f "${path}" ]]; then
            echo "Downloaded file from ${url} is not a valid ${kind} archive." >&2
            file "${path}" >&2 || true
            rm -f "${path}"
        fi
    done

    echo "Failed to download a valid ${kind} archive for ${name}." >&2
    echo "Tried mirrors: ${chime6_mirrors[*]}" >&2
    exit 1
}

is_valid_archive() {
    local path=$1
    local kind=$2
    if [[ "${kind}" == "tar" ]]; then
        tar -tzf "${path}" >/dev/null 2>&1
    else
        tar -tzf "${path}" >/dev/null 2>&1 || gzip -t "${path}" >/dev/null 2>&1
    fi
}

extract_once() {
    local name=$1
    local kind=${2:-tar}
    local marker="${raw_dir}/.${name}.extracted"
    if [[ -f "${marker}" ]]; then
        echo "${name} already extracted. Skipping extraction."
        return
    fi

    echo "Extracting ${raw_dir}/${name}"
    if tar -xzf "${raw_dir}/${name}" -C "${raw_dir}"; then
        touch "${marker}"
        return
    fi

    if [[ "${kind}" != "gzip" ]]; then
        echo "Failed to extract tar.gz archive: ${raw_dir}/${name}" >&2
        exit 1
    fi

    "${python_bin}" - "${raw_dir}/${name}" "${raw_dir}" <<'PY'
import gzip
import shutil
import sys
import tarfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
tmp = out_dir / archive.name.replace(".gz", "")

with gzip.open(archive, "rb") as src, tmp.open("wb") as dst:
    shutil.copyfileobj(src, dst)

if tarfile.is_tarfile(tmp):
    with tarfile.open(tmp) as tar:
        tar.extractall(out_dir)
    tmp.unlink()
else:
    target = out_dir / tmp.name
    if tmp != target:
        shutil.move(str(tmp), str(target))
PY
    touch "${marker}"
}

download CHiME6_eval.tar.gz tar
download CHiME6_transcriptions.tar.gz gzip
extract_once CHiME6_eval.tar.gz tar
extract_once CHiME6_transcriptions.tar.gz gzip

echo "Writing CHiME-6 eval manifest to ${manifest_path}"
"${python_bin}" - \
    "${raw_dir}" \
    "${processed_dir}" \
    "${manifest_path}" \
    "${array}" \
    "${channel}" \
    "${max_items}" <<'PY'
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

raw_dir = Path(sys.argv[1]).resolve()
processed_dir = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3]).resolve()
array = sys.argv[4]
channel = sys.argv[5]
max_items = int(sys.argv[6])
sample_rate = 16000
sessions = ("S01", "S21")


def seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    parts = value.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(value)


def normalize_text(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_one(patterns):
    for pattern in patterns:
        matches = sorted(raw_dir.rglob(pattern))
        if matches:
            return matches[0].resolve()
    return None


def find_transcript(session):
    return find_one(
        [
            f"transcriptions/eval/{session}.json",
            f"**/transcriptions/eval/{session}.json",
            f"**/{session}.json",
        ]
    )


def find_array_audio(session, segment):
    ref = str(segment.get("ref") or "").strip()
    array_candidates = [ref, array] if ref else [array]
    patterns = []
    for array_id in dict.fromkeys(item for item in array_candidates if item):
        patterns.extend(
            [
                f"audio/eval/**/{session}_{array_id}.{channel}.wav",
                f"**/audio/eval/**/{session}_{array_id}.{channel}.wav",
                f"**/{session}_{array_id}.{channel}.wav",
                f"audio/eval/**/{session}_{array_id}.wav",
                f"**/audio/eval/**/{session}_{array_id}.wav",
                f"**/{session}_{array_id}.wav",
            ]
        )
    return find_one(patterns)


def find_worn_audio(session, speaker):
    return find_one(
        [
            f"audio/eval/**/{session}_{speaker}.wav",
            f"**/audio/eval/**/{session}_{speaker}.wav",
            f"**/{session}_{speaker}.wav",
        ]
    )


def read_clip(path, start, end):
    with sf.SoundFile(str(path)) as f:
        if f.samplerate != sample_rate:
            raise RuntimeError(f"Expected {sample_rate} Hz audio, got {f.samplerate}: {path}")
        start_frame = max(0, int(round(start * f.samplerate)))
        end_frame = min(len(f), int(round(end * f.samplerate)))
        if end_frame <= start_frame:
            raise RuntimeError(f"Empty clip for {path}: {start}-{end}")
        f.seek(start_frame)
        audio = f.read(end_frame - start_frame, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32)


def write_clip(path, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def safe_id(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


segments_by_session = {}
for session in sessions:
    transcript_path = find_transcript(session)
    if transcript_path is None:
        raise FileNotFoundError(f"Missing CHiME-6 eval transcript for {session} under {raw_dir}")
    with transcript_path.open() as f:
        segments_by_session[session] = json.load(f)

enrollment_candidates = defaultdict(list)
for session, segments in segments_by_session.items():
    for idx, segment in enumerate(segments):
        speaker = segment["speaker"]
        worn = find_worn_audio(session, speaker)
        if worn is None:
            continue
        start = seconds(segment["start_time"])
        end = seconds(segment["end_time"])
        if end <= start:
            continue
        enrollment_candidates[(session, speaker)].append((idx, start, end, worn))

manifest_rows = []
skipped = 0
for session, segments in segments_by_session.items():
    for idx, segment in enumerate(segments):
        speaker = segment["speaker"]
        start = seconds(segment["start_time"])
        end = seconds(segment["end_time"])
        if end <= start:
            skipped += 1
            continue

        mix_audio_path = find_array_audio(session, segment)
        enrollment = next(
            (item for item in enrollment_candidates[(session, speaker)] if item[0] != idx),
            None,
        )
        if mix_audio_path is None or enrollment is None:
            skipped += 1
            continue

        item_id = safe_id(f"{session}_{idx:05d}_{speaker}_{segment.get('start_time')}_{segment.get('end_time')}")
        mix_out = processed_dir / "mix" / f"{item_id}.wav"
        enrollment_out = processed_dir / "enrollment" / f"{item_id}.wav"

        write_clip(mix_out, read_clip(mix_audio_path, start, end))
        _, enr_start, enr_end, enr_path = enrollment
        write_clip(enrollment_out, read_clip(enr_path, enr_start, enr_end))

        manifest_rows.append(
            {
                "id": item_id,
                "category": "system",
                "mix": str(mix_out.resolve()),
                "enrollment": str(enrollment_out.resolve()),
                "transcript": normalize_text(segment["words"]),
                "session": session,
                "speaker": speaker,
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "array": str(segment.get("ref") or array),
                "channel": channel,
                "enrollment_recording": str(enr_path),
                "mix_recording": str(mix_audio_path),
            }
        )
        if max_items > 0 and len(manifest_rows) >= max_items:
            break
    if max_items > 0 and len(manifest_rows) >= max_items:
        break

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", newline="") as f:
    fieldnames = [
        "id",
        "category",
        "mix",
        "enrollment",
        "transcript",
        "session",
        "speaker",
        "start_time",
        "end_time",
        "array",
        "channel",
        "enrollment_recording",
        "mix_recording",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Wrote {len(manifest_rows)} rows to {manifest_path}")
if skipped:
    print(f"Skipped {skipped} segments without valid audio/enrollment", file=sys.stderr)
PY

echo "Done."
echo "Manifest: ${manifest_path}"
echo "Run evaluation with:"
echo "  ./run.sh --stage 7 --eval_config CHiME6/default"
