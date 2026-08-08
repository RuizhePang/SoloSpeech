#!/usr/bin/env python3

# This code is adapted from SpeakerBeam:
# https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/create_enrollment_csv_all.py

import csv
import sys
from collections import defaultdict


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <mixture_csv> <out_enrollment_csv>", file=sys.stderr)
        sys.exit(1)

    mix_csv = sys.argv[1]
    out_enr_csv = sys.argv[2]

    spk2utts = defaultdict(set)
    utt2pathlen = {}
    mix_ids = []

    with open(mix_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            mix_id, _, s1_path, s2_path, _, length = row
            mix_ids.append(mix_id)
            utt1id, utt2id = mix_id.split("_")
            spk1, spk2 = utt1id.split("-")[0], utt2id.split("-")[0]
            spk2utts[spk1].add(utt1id)
            spk2utts[spk2].add(utt2id)
            utt2pathlen[utt1id] = (s1_path, length)
            utt2pathlen[utt2id] = (s2_path, length)

    with open(out_enr_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mixture_id", "utterance_id", "enr_path1", "length1", "enr_path2", "length2", "..."])
        for mix_id in mix_ids:
            utt1, utt2 = mix_id.split("_")

            for utt in (utt1, utt2):
                row = [mix_id, utt]
                for utt_id in sorted(spk2utts[utt.split("-")[0]]):
                    if utt_id == utt:
                        continue
                    enr_utt, length = utt2pathlen[utt_id]
                    row.extend([enr_utt, length])
                writer.writerow(row)


if __name__ == "__main__":
    main()
