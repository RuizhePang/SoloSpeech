#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <out_dir> <python_bin> [mic] [max_items]" >&2
    echo "Example: $0 data/train/LibriCSS python 0 0" >&2
    exit 1
fi

out_dir=$1
python_bin=$2
mic=${3:-0}
max_items=${4:-0}

repo_url=${LIBRICSS_REPO_URL:-"https://github.com/chenzhuo1011/libri_css.git"}
drive_file_id=${LIBRICSS_DRIVE_FILE_ID:-"1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l"}
repo_dir=${LIBRICSS_REPO_DIR:-"${out_dir}/libri_css"}
raw_dir="${out_dir}/raw"
zip_path="${raw_dir}/for_release.zip"
release_dir="${raw_dir}/for_release"
processed_dir="${out_dir}/test"
mono_dir="${processed_dir}/monaural"
manifest_path="${processed_dir}/manifest.csv"
speaker_info="${repo_dir}/speaker_enrollment/libricss_speaker_info.jsonl"

mkdir -p "${out_dir}" "${raw_dir}" "${processed_dir}"

for tool in git unzip wget; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "${tool} is required by LibriCSS preparation." >&2
        exit 1
    fi
done

"${python_bin}" - <<'PY'
import importlib.util
import sys

missing = [module for module in ("soundfile", "numpy", "tqdm") if importlib.util.find_spec(module) is None]
if missing:
    print("Missing Python packages required by LibriCSS preparation: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

if [[ -d "${repo_dir}/.git" ]]; then
    git -C "${repo_dir}" fetch --depth 1 origin master
    git -C "${repo_dir}" checkout -q FETCH_HEAD
else
    git clone --depth 1 "${repo_url}" "${repo_dir}"
fi

is_valid_zip() {
    [[ -f "$1" ]] && unzip -tq "$1" >/dev/null 2>&1
}

download_with_gdown() {
    "${python_bin}" - "$drive_file_id" "$zip_path" <<'PY'
import importlib.util
import subprocess
import sys

file_id, output_path = sys.argv[1], sys.argv[2]
if importlib.util.find_spec("gdown") is None:
    sys.exit(2)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "gdown",
        "--fuzzy",
        f"https://drive.google.com/file/d/{file_id}/view",
        "-O",
        output_path,
    ]
)
PY
}

ensure_gdown() {
    if "${python_bin}" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("gdown") is not None else 1)
PY
    then
        return 0
    fi

    echo "Installing gdown into ${python_bin}'s environment"
    "${python_bin}" -m pip install -q gdown
}

download_with_wget() {
    (
        cd "${raw_dir}"
        rm -f /tmp/libricss_cookies.txt /tmp/libricss_download.html
        wget --quiet --save-cookies /tmp/libricss_cookies.txt --keep-session-cookies --no-check-certificate \
            "https://drive.google.com/uc?export=download&id=${drive_file_id}" \
            -O /tmp/libricss_download.html
        confirm=$(sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' /tmp/libricss_download.html | head -n 1)
        uuid=$(sed -rn 's/.*uuid=([0-9A-Za-z_-]+).*/\1/p' /tmp/libricss_download.html | head -n 1)
        if [[ -n "${uuid}" ]]; then
            url="https://drive.usercontent.google.com/download?id=${drive_file_id}&export=download&confirm=t&uuid=${uuid}"
        elif [[ -n "${confirm}" ]]; then
            url="https://drive.google.com/uc?export=download&confirm=${confirm}&id=${drive_file_id}"
        else
            url="https://drive.google.com/uc?export=download&id=${drive_file_id}"
        fi
        wget --load-cookies /tmp/libricss_cookies.txt --no-check-certificate "${url}" \
            -O for_release.zip
        rm -f /tmp/libricss_cookies.txt /tmp/libricss_download.html
    )
}

if ! is_valid_zip "${zip_path}"; then
    if [[ -f "${zip_path}" ]]; then
        echo "${zip_path} is not a valid zip file. Removing it before retrying." >&2
        rm -f "${zip_path}"
    fi
    echo "Downloading LibriCSS for_release.zip into ${raw_dir}"
    if ! ensure_gdown || ! download_with_gdown; then
        echo "gdown is unavailable or failed; falling back to wget." >&2
        download_with_wget
    fi
else
    echo "${zip_path} already exists and is valid. Skipping download."
fi

if ! is_valid_zip "${zip_path}"; then
    echo "Failed to download a valid LibriCSS zip file: ${zip_path}" >&2
    echo "Install gdown in ${python_bin}'s environment, or manually download the official LibriCSS file to that path." >&2
    echo "Official file: https://drive.google.com/file/d/${drive_file_id}/view" >&2
    exit 1
fi

if [[ ! -d "${release_dir}" ]]; then
    echo "Extracting ${zip_path}"
    unzip -q "${zip_path}" -d "${raw_dir}"
else
    echo "${release_dir} already exists. Skipping extraction."
fi

if [[ -f "${release_dir}/segment_libricss.py" ]]; then
    echo "Running LibriCSS segmentation"
    (
        cd "${release_dir}"
        "${python_bin}" segment_libricss.py -data_path .
    )
else
    echo "Missing ${release_dir}/segment_libricss.py" >&2
    exit 1
fi

echo "Preparing monaural LibriCSS utterance data with mic ${mic}"
"${python_bin}" "${repo_dir}/dataprep/python/dataprep.py" \
    --srcpath "${release_dir}" \
    --tgtpath "${mono_dir}" \
    --mics "${mic}"

echo "Writing manifest to ${manifest_path}"
"${python_bin}" - \
    "${release_dir}" \
    "${mono_dir}" \
    "${speaker_info}" \
    "${manifest_path}" \
    "${max_items}" <<'PY'
import csv
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

release_dir = Path(sys.argv[1]).resolve()
mono_dir = Path(sys.argv[2]).resolve()
speaker_info_path = Path(sys.argv[3]).resolve()
manifest_path = Path(sys.argv[4]).resolve()
max_items = int(sys.argv[5])


def normalize_text(text):
    return " ".join(text.strip().split())


def wav_key(path):
    stem = Path(path).stem
    matches = re.findall(r"\d+-\d+-\d+", stem)
    return matches[-1] if matches else stem


def read_session_transcripts():
    rows = []
    for meeting_info in sorted(release_dir.glob("*/*/transcription/meeting_info.txt")):
        session_dir = meeting_info.parents[1]
        session = session_dir.name
        condition = session_dir.parent.name
        with meeting_info.open() as f:
            next(f, None)
            for idx, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                start, end, spk, utt_id, text = parts[:5]
                utt_stem = f"utterance_{idx}"
                rows.append(
                    {
                        "condition": condition,
                        "session": session,
                        "utt_stem": utt_stem,
                        "speaker": spk,
                        "utt_id": utt_id,
                        "transcript": normalize_text(text),
                    }
                )
    return rows


def index_wavs(root):
    by_key = defaultdict(list)
    for path in sorted(root.rglob("*.wav")):
        by_key[wav_key(path)].append(path.resolve())
    return by_key


def first_existing(*paths):
    for path in paths:
        if path is not None and Path(path).is_file():
            return Path(path).resolve()
    return None


def find_mixture(row):
    return first_existing(
        mono_dir / "utterances" / row["session"] / f"{row['utt_stem']}.wav",
        mono_dir / "utterances" / row["session"] / f"{row['utt_id']}.wav",
    )


def find_by_utt(index, utt_id):
    for path in index.get(utt_id, []):
        return path
    return None


rows = read_session_transcripts()
all_wavs = index_wavs(release_dir)
mono_wavs = index_wavs(mono_dir / "utterances")

speaker_info = {}
if speaker_info_path.is_file():
    with speaker_info_path.open() as f:
        for line in f:
            item = json.loads(line)
            speaker_info[item["dataid"]] = item

session_rows = defaultdict(list)
for row in rows:
    session_rows[row["session"]].append(row)

session_enrollments = defaultdict(lambda: defaultdict(deque))
for session, item in speaker_info.items():
    for spk, utt_ids in item.get("unused_uttid", {}).items():
        for utt_id in utt_ids:
            path = find_by_utt(all_wavs, utt_id) or find_by_utt(mono_wavs, utt_id)
            if path is not None:
                session_enrollments[session][spk].append(path)

for session, items in session_rows.items():
    for row in items:
        path = find_by_utt(all_wavs, row["utt_id"]) or find_mixture(row)
        if path is not None:
            session_enrollments[session][row["speaker"]].append(path)

manifest_rows = []
skipped = 0
for row in rows:
    mixture = find_mixture(row)
    if mixture is None:
        skipped += 1
        continue

    spk = row["speaker"]
    source = find_by_utt(all_wavs, row["utt_id"]) or mixture
    enrollment = None
    for candidate in session_enrollments[row["session"]][spk]:
        if wav_key(candidate) != row["utt_id"]:
            enrollment = candidate
            break
    if enrollment is None:
        skipped += 1
        continue

    manifest_rows.append(
        {
            "id": f"{row['condition']}_{row['session']}_{row['utt_stem']}_{row['utt_id']}",
            "category": "all",
            "mix": str(mixture),
            "source": str(source),
            "enrollment": str(enrollment),
            "transcript": row["transcript"],
            "condition": row["condition"],
            "session": row["session"],
            "speaker": spk,
            "utt_id": row["utt_id"],
        }
    )
    if max_items > 0 and len(manifest_rows) >= max_items:
        break

manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", newline="") as f:
    fieldnames = ["id", "category", "mix", "source", "enrollment", "transcript", "condition", "session", "speaker", "utt_id"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Wrote {len(manifest_rows)} rows to {manifest_path}")
if skipped:
    print(f"Skipped {skipped} rows without mixture or enrollment", file=sys.stderr)
PY

echo "Done."
echo "Manifest: ${manifest_path}"
echo "Run evaluation with:"
echo "  ./run.sh --stage 7 --eval_config default --test_dir ${out_dir} --test_manifest ${manifest_path}"
