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
repo_dir=${LIBRICSS_REPO_DIR:-"${out_dir}/libri_css"}
raw_dir="${out_dir}/raw"
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

if [[ ! -f "${raw_dir}/for_release.zip" ]]; then
    echo "Downloading LibriCSS for_release.zip into ${raw_dir}"
    (
        cd "${raw_dir}"
        wget --load-cookies /tmp/libricss_cookies.txt \
            "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/libricss_cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l" \
            -O for_release.zip
        rm -f /tmp/libricss_cookies.txt
    )
else
    echo "${raw_dir}/for_release.zip already exists. Skipping download."
fi

if [[ ! -d "${release_dir}" ]]; then
    echo "Extracting ${raw_dir}/for_release.zip"
    unzip -q "${raw_dir}/for_release.zip" -d "${raw_dir}"
else
    echo "${release_dir} already exists. Skipping extraction."
fi

if [[ -f "${release_dir}/segment_libricss.py" && ! -f "${release_dir}/all_res.json" ]]; then
    echo "Running LibriCSS segmentation"
    (
        cd "${release_dir}"
        "${python_bin}" segment_libricss.py -data_path .
    )
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
