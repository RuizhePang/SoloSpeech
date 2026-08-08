#!/usr/bin/env python3

# This code is adapted from SpeakerBeam:
# https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/create_enrollment_csv_fixed.py

import csv
import sys


def main():
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <mixture_csv> <map_mixture2enrollment> <out_enrollment_csv>",
            file=sys.stderr,
        )
        sys.exit(1)

    mix_csv = sys.argv[1]
    map_mix2enroll = sys.argv[2]
    out_enr_csv = sys.argv[3]

    utt2pathlen = {}
    mix_ids = []

    with open(mix_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            mix_id, _, s1_path, s2_path, _, length = row
            mix_ids.append(mix_id)
            utt2pathlen[f"s1/{mix_id}"] = (s1_path, length)
            utt2pathlen[f"s2/{mix_id}"] = (s2_path, length)

    mix2enroll = {}
    with open(map_mix2enroll) as f:
        for line in f:
            mix_id, utt_id, enroll_id = line.strip().split()
            mix2enroll[(mix_id, utt_id)] = enroll_id

    with open(out_enr_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mixture_id", "utterance_id", "enr_path1", "length1"])
        for mix_id in mix_ids:
            utt1, utt2 = mix_id.split("_")

            for utt in (utt1, utt2):
                enr_id = mix2enroll[(mix_id, utt)]
                enr_utt, length = utt2pathlen[enr_id]
                writer.writerow([mix_id, utt, enr_utt, length])


if __name__ == "__main__":
    main()
