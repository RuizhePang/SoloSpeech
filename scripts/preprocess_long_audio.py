#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diarization-based chunking + enroll segment selection (no VAD).

Rules:

1. Diarization:
   - Run pyannote diarization.
   - For each speaker, merge consecutive regions to avoid over-fragmentation.
   - BUT: when exporting per-chunk diarization_segments, DO NOT MERGE.
     We output raw diarization segments for higher timestamp precision.

2. Chunking:
   - Using the merged diarization, build a global time grid.
   - Find intervals where no speaker is active and duration >= 0.8s.
   - Use the midpoint of such intervals as preferred cut points.
   - From t=0, sequentially build chunks:
       * Try to place each chunk end so that the length is in [min_chunk_len, max_chunk_len]
         and cut at a silence midpoint if possible.
       * If no silence midpoint in that range, cut at start + max_chunk_len (or end of file).
       * If the last chunk is shorter than min_chunk_len, merge it into the previous one.

3. Enroll regions per speaker:
   - Build a global time grid; for each interval [t_i, t_{i+1}):
       * Let A be the active speakers.
       * If len(A) == 1: this interval is "pure" for that speaker.
       * If len(A) >= 2: this interval is "overlap" for every speaker in A.
   - For each speaker:
       * Merge close pure intervals -> pure_regions_by_speaker[spk]
       * Merge close overlap intervals -> overlap_regions_by_speaker[spk]
       * merged speech regions (from step 1) -> speech_regions_by_speaker[spk]

   - From pure regions:
       * Extract 1.5–4s segments as "valid enroll" (global_valid_enroll_by_speaker[spk]).
         Longer regions are center-cropped to 4s.

   - From overlap regions:
       * Extract 1.5–4s segments as "backup enroll" (global_backup_enroll_by_speaker[spk]).

   - From speech regions:
       * Extract "invalid enroll" segments: any speech region (no minimum length),
         center-cropped to <= max_enroll_len when necessary
         (global_invalid_enroll_by_speaker[spk]).
       * This guarantees each speaker will always have at least one enroll candidate
         as long as they speak somewhere.

4. Per-chunk enroll assignment:

   For each chunk, for each speaker (all speakers):

   - diarization_segments: RAW diarization segments (unmerged) inside the chunk (can be empty).

   - enroll_segments priority (to collect candidates):
       1) valid enroll in this chunk (pure & length in [1.5, 4])
       2) global valid enroll (pure from anywhere)
       3) backup enroll in this chunk (overlap & length in [1.5, 4])
       4) global backup enroll (overlap from anywhere)
       5) invalid enroll in this chunk (speech from merged regions)
       6) global invalid enroll (speech from anywhere)

   - Then:
       * Take up to `max_enroll_per_speaker_per_chunk` candidates in that priority order.
       * If total < `max_enroll_per_speaker_per_chunk`, repeat (cycle) existing ones until we have exactly that many.

Output JSON:
{
  "audio_path": ...,
  "chunks": [
    {
      "start": float,
      "end": float,
      "speakers": {
        "SPEAKER_00": {
          "enroll_segments": [
            {"start": s1, "end": e1, "type": "valid"},
            ...
          ],
          "diarization_segments": [[d1_s, d1_e], ...]   # RAW, not merged
        },
        ...
      }
    },
    ...
  ],
  "global_enroll": {
    "SPEAKER_00": [
      {"start": ..., "end": ..., "type": "valid"},
      {"start": ..., "end": ..., "type": "backup"},
      {"start": ..., "end": ..., "type": "invalid"}
    ],
    ...
  }
}
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

import torchaudio
from pyannote.audio import Pipeline


# =========================
# Basic data structure
# =========================

@dataclass
class Segment:
    """Simple segment representation."""
    start: float
    end: float
    speaker: str  # speaker_id


# =========================
# 1. Diarization
# =========================

def run_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    model_name: str = "pyannote/speaker-diarization-3.1",
) -> List[Segment]:
    """
    Run pyannote diarization and return a list of Segment objects (raw output).
    """
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        raise ValueError("Please provide a HuggingFace token via --hf_token or HF_TOKEN env var.")

    pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    diarization = pipeline(audio_path, num_speakers=num_speakers)

    segments: List[Segment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            Segment(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            )
        )
    segments.sort(key=lambda x: (x.start, x.end))
    return segments


# =========================
# 2. Utilities
# =========================

def get_audio_duration(audio_path: str) -> float:
    """Return audio duration in seconds."""
    info = torchaudio.info(audio_path)
    num_frames = info.num_frames
    sr = info.sample_rate
    return float(num_frames) / float(sr)


def group_segments_by_speaker(segments: List[Segment]) -> Dict[str, List[Segment]]:
    """Group segments by speaker_id (keeps input segments as-is, sorted)."""
    speakers: Dict[str, List[Segment]] = {}
    for seg in segments:
        speakers.setdefault(seg.speaker, []).append(seg)
    for spk in speakers:
        speakers[spk].sort(key=lambda x: (x.start, x.end))
    return speakers


def merge_close_segments(
    segments: List[Segment],
    gap_threshold: float = 0.2,
    min_duration: float = 0.0,
) -> List[Segment]:
    """
    Merge temporally close segments for the same speaker.

    gap_threshold: if the time gap between two consecutive segments is less,
                   they will be merged.
    min_duration: segments shorter than this will be dropped.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: (x.start, x.end))
    merged: List[Segment] = []
    cur = segments[0]

    for seg in segments[1:]:
        if seg.speaker == cur.speaker and seg.start - cur.end <= gap_threshold:
            cur = Segment(start=cur.start, end=max(cur.end, seg.end), speaker=cur.speaker)
        else:
            if cur.end - cur.start >= min_duration:
                merged.append(cur)
            cur = seg

    if cur.end - cur.start >= min_duration:
        merged.append(cur)

    return merged


def get_diarization_segments_in_chunk_for_speaker(
    chunk_start: float,
    chunk_end: float,
    segments_for_speaker: List[Segment],
    min_overlap: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Return diarization segments for a given speaker inside the chunk,
    clipped to [chunk_start, chunk_end].

    NOTE:
      - If you pass RAW diarization segments here, output will be RAW (unmerged).
      - If you pass merged speech regions, output will be merged.
    """
    diarization_segments: List[Tuple[float, float]] = []
    for seg in segments_for_speaker:
        s = max(seg.start, chunk_start)
        e = min(seg.end, chunk_end)
        if e - s >= min_overlap:
            diarization_segments.append((float(s), float(e)))
    return diarization_segments


# =========================
# 3. Global time grid & silence
# =========================

def build_time_grid(segments: List[Segment], audio_duration: float) -> List[float]:
    """
    Build a sorted list of unique time boundaries from:
      - 0
      - audio_duration
      - all segment starts and ends.
    """
    boundaries = [0.0, audio_duration]
    for seg in segments:
        boundaries.append(seg.start)
        boundaries.append(seg.end)
    boundaries = sorted(set(boundaries))
    return boundaries

def build_chunks_from_silence_avoid_speech(
    audio_duration: float,
    silence_intervals: List[Tuple[float, float]],
    raw_segments_all: List[Segment], 
    min_chunk_len: float = 5.0,
    max_chunk_len: float = 10.0,
    guard_eps: float = 0.02,
) -> List[Tuple[float, float]]:
    """
    Build chunks sequentially from t=0 using silence midpoints as preferred cut points,
    BUT avoid cutting inside any RAW diarization speech region.
    """

    # precompute silence midpoints
    silence_midpoints = sorted([(s + e) / 2.0 for (s, e) in silence_intervals])

    # precompute raw boundaries (all starts/ends)
    raw_boundaries = [0.0, audio_duration]
    for seg in raw_segments_all:
        raw_boundaries.append(float(seg.start))
        raw_boundaries.append(float(seg.end))
    raw_boundaries = sorted(set(raw_boundaries))

    def is_in_speech(t: float) -> bool:
        # if t is strictly inside any speech segment (with eps margin), return True
        for seg in raw_segments_all:
            if (seg.start + guard_eps) < t < (seg.end - guard_eps):
                return True
        return False

    def nearest_safe_boundary(t: float, lo: float, hi: float) -> Optional[float]:
        """
        Find a boundary in [lo, hi] close to t that is NOT inside speech.
        If none, return None.
        """
        cands = [b for b in raw_boundaries if lo <= b <= hi]
        if not cands:
            return None
        safe = [b for b in cands if not is_in_speech(b)]
        if not safe:
            return None
        return min(safe, key=lambda x: abs(x - t))

    def nearest_boundary_any(t: float, lo: float, hi: float) -> Optional[float]:
        cands = [b for b in raw_boundaries if lo <= b <= hi]
        if not cands:
            return None
        return min(cands, key=lambda x: abs(x - t))

    chunks: List[Tuple[float, float]] = []
    cur_start = 0.0
    target_center_offset = 0.5 * (min_chunk_len + max_chunk_len)

    while cur_start < audio_duration - 1e-6:
        remaining = audio_duration - cur_start
        if remaining <= min_chunk_len:
            chunks.append((cur_start, audio_duration))
            break

        target_min = cur_start + min_chunk_len
        target_max = min(cur_start + max_chunk_len, audio_duration)

        # 1) pick a proposed end based on silence midpoint if exists
        candidates = [m for m in silence_midpoints if (target_min <= m <= target_max)]
        if candidates:
            target = cur_start + target_center_offset
            proposed_end = min(candidates, key=lambda x: abs(x - target))
            proposed_end = min(max(proposed_end, target_min), target_max)
        else:
            proposed_end = target_max

        # 2) guard: avoid cutting inside speech
        end = proposed_end
        if is_in_speech(end):
            # try to find a safe boundary within window
            target = cur_start + target_center_offset
            safe_b = nearest_safe_boundary(target, target_min, target_max)
            if safe_b is not None:
                end = safe_b
            else:
                # if no safe boundary, at least snap to nearest raw boundary to reduce cutting mid-phoneme
                any_b = nearest_boundary_any(target, target_min, target_max)
                if any_b is not None:
                    end = any_b
                else:
                    end = proposed_end  # fallback

        # ensure progress
        if end <= cur_start + 1e-4:
            end = min(cur_start + max_chunk_len, audio_duration)

        chunks.append((cur_start, end))
        cur_start = end

    # merge last short chunk
    if len(chunks) >= 2:
        last_start, last_end = chunks[-1]
        if last_end - last_start < min_chunk_len:
            prev_start, _ = chunks[-2]
            chunks[-2] = (prev_start, last_end)
            chunks.pop()

    return [(float(s), float(e)) for (s, e) in chunks]


def compute_silence_intervals_from_diarization(
    segments: List[Segment],
    audio_duration: float,
    min_silence_len: float = 0.8,
) -> List[Tuple[float, float]]:
    """
    Using diarization only, compute intervals where no speakers are active
    and duration >= min_silence_len.
    """
    if not segments:
        if audio_duration >= min_silence_len:
            return [(0.0, audio_duration)]
        else:
            return []

    boundaries = build_time_grid(segments, audio_duration)
    silence_intervals: List[Tuple[float, float]] = []
    n = len(boundaries)

    for i in range(n - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start
        if dur <= 0:
            continue

        active: Set[str] = set()
        for seg in segments:
            if not (seg.end <= start or seg.start >= end):
                active.add(seg.speaker)

        if len(active) == 0 and dur >= min_silence_len:
            silence_intervals.append((start, end))

    return silence_intervals


# =========================
# 4. Pure / overlap regions
# =========================

def compute_pure_and_overlap_regions(
    segments: List[Segment],
    audio_duration: float,
    min_region_len: float = 0.1,
) -> Tuple[Dict[str, List[Segment]], Dict[str, List[Segment]]]:
    """
    Compute pure (non-overlap) and overlap regions for each speaker.

    For each interval [t_i, t_{i+1}) in the global time grid:
      - active speakers A:
          * if len(A) == 1: this interval is pure for that speaker.
          * if len(A) >= 2: this interval is overlap for every speaker in A.
    """
    pure_by_speaker: Dict[str, List[Segment]] = {}
    overlap_by_speaker: Dict[str, List[Segment]] = {}

    if not segments:
        return pure_by_speaker, overlap_by_speaker

    boundaries = build_time_grid(segments, audio_duration)
    n = len(boundaries)

    for i in range(n - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start
        if dur <= 0:
            continue

        active: List[str] = []
        for seg in segments:
            if not (seg.end <= start or seg.start >= end):
                active.append(seg.speaker)

        if not active:
            continue

        unique_speakers = set(active)
        if len(unique_speakers) == 1:
            spk = next(iter(unique_speakers))
            pure_by_speaker.setdefault(spk, []).append(Segment(start, end, spk))
        else:
            for spk in unique_speakers:
                overlap_by_speaker.setdefault(spk, []).append(Segment(start, end, spk))

    for spk, segs in pure_by_speaker.items():
        pure_by_speaker[spk] = merge_close_segments(
            segs, gap_threshold=0.2, min_duration=min_region_len
        )

    for spk, segs in overlap_by_speaker.items():
        overlap_by_speaker[spk] = merge_close_segments(
            segs, gap_threshold=0.2, min_duration=min_region_len
        )

    return pure_by_speaker, overlap_by_speaker


# =========================
# 5. Enroll segment helpers
# =========================

def pick_valid_or_backup_from_regions(
    segs: List[Segment],
    min_len: float = 1.5,
    max_len: float = 4.0,
    max_num: int = 10,
) -> List[Tuple[float, float]]:
    """
    From pure/overlap regions, pick enroll segments of length in [min_len, max_len].
    """
    import random

    candidates: List[Tuple[float, float]] = []

    for seg in segs:
        dur = seg.end - seg.start
        if dur < min_len:
            continue
        if dur <= max_len:
            candidates.append((seg.start, seg.end))
        else:
            center = (seg.start + seg.end) / 2.0
            start = center - max_len / 2.0
            end = start + max_len
            start = max(start, seg.start)
            end = min(end, seg.end)
            candidates.append((start, end))

    random.shuffle(candidates)
    return candidates[:max_num]


def pick_valid_or_backup_in_chunk(
    chunk_start: float,
    chunk_end: float,
    segs_for_speaker: List[Segment],
    min_len: float = 1.5,
    max_len: float = 4.0,
    max_num: int = 3,
) -> List[Tuple[float, float]]:
    """
    From pure/overlap regions, pick enroll segments inside a chunk with length in [min_len, max_len].
    """
    import random

    candidates: List[Tuple[float, float]] = []

    for seg in segs_for_speaker:
        s = max(seg.start, chunk_start)
        e = min(seg.end, chunk_end)
        dur = e - s
        if dur < min_len:
            continue
        if dur <= max_len:
            candidates.append((s, e))
        else:
            center = (s + e) / 2.0
            start = center - max_len / 2.0
            end = start + max_len
            start = max(start, s)
            end = min(end, e)
            candidates.append((start, end))

    random.shuffle(candidates)
    return candidates[:max_num]


def pick_invalid_from_speech_regions(
    segs: List[Segment],
    max_len: float = 4.0,
    max_num: int = 3,
) -> List[Tuple[float, float]]:
    """
    From general speech regions (merged speech), pick "invalid" enroll segments.

    Rules:
      - No minimum length: any region with dur > 0 is usable.
      - If dur <= max_len: take the whole region.
      - If dur > max_len: take a centered subsegment of length max_len.
    """
    import random

    candidates: List[Tuple[float, float]] = []

    for seg in segs:
        dur = seg.end - seg.start
        if dur <= 0:
            continue
        if dur <= max_len:
            candidates.append((seg.start, seg.end))
        else:
            center = (seg.start + seg.end) / 2.0
            start = center - max_len / 2.0
            end = start + max_len
            start = max(start, seg.start)
            end = min(end, seg.end)
            candidates.append((start, end))

    random.shuffle(candidates)
    return candidates[:max_num]


def pick_invalid_in_chunk_from_speech(
    chunk_start: float,
    chunk_end: float,
    segs_for_speaker: List[Segment],
    max_len: float = 4.0,
    max_num: int = 3,
) -> List[Tuple[float, float]]:
    """
    From general speech regions (merged speech), pick "invalid" enroll segments within a chunk.
    """
    import random

    candidates: List[Tuple[float, float]] = []

    for seg in segs_for_speaker:
        s = max(seg.start, chunk_start)
        e = min(seg.end, chunk_end)
        dur = e - s
        if dur <= 0:
            continue
        if dur <= max_len:
            candidates.append((s, e))
        else:
            center = (s + e) / 2.0
            start = center - max_len / 2.0
            end = start + max_len
            start = max(start, s)
            end = min(end, e)
            candidates.append((start, end))

    random.shuffle(candidates)
    return candidates[:max_num]


# =========================
# 7. Main processing
# =========================

def process_audio_with_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_chunk_len: float = 5.0,
    max_chunk_len: float = 10.0,
    min_enroll_len: float = 1.5,
    max_enroll_len: float = 4.0,
    max_enroll_per_speaker_global: int = 10,
    max_enroll_per_speaker_per_chunk: int = 8,
    min_silence_len: float = 0.8,
):
    """
    Main pipeline that follows the specified rules.
    """

    # 1) Diarization (RAW)
    raw_segments = run_diarization(audio_path, hf_token=hf_token, num_speakers=num_speakers)
    audio_duration = get_audio_duration(audio_path)

    # Group RAW by speaker (for exporting per-chunk diarization_segments WITHOUT merging)
    raw_segments_by_speaker = group_segments_by_speaker(raw_segments)

    # 2) Merge consecutive regions for each speaker (for chunking/pure/overlap/enroll only)
    speech_regions_by_speaker: Dict[str, List[Segment]] = {}
    for spk, segs in raw_segments_by_speaker.items():
        speech_regions_by_speaker[spk] = merge_close_segments(
            segs, gap_threshold=0.1, min_duration=0.0
        )

    # Flatten merged segments back into a list
    merged_segments: List[Segment] = []
    for segs in speech_regions_by_speaker.values():
        merged_segments.extend(segs)
    merged_segments.sort(key=lambda x: (x.start, x.end))

    speakers_all = set(speech_regions_by_speaker.keys())

    # 3) Silence intervals from MERGED diarization
    silence_intervals = compute_silence_intervals_from_diarization(
        merged_segments,
        audio_duration=audio_duration,
        min_silence_len=min_silence_len,
    )

    # 4) Pure & overlap regions from MERGED segments
    pure_regions_by_speaker, overlap_regions_by_speaker = compute_pure_and_overlap_regions(
        merged_segments,
        audio_duration=audio_duration,
        min_region_len=0.1,
    )

    # 5) Global enroll candidates per speaker
    global_valid_enroll_by_speaker: Dict[str, List[Tuple[float, float]]] = {}
    global_backup_enroll_by_speaker: Dict[str, List[Tuple[float, float]]] = {}
    global_invalid_enroll_by_speaker: Dict[str, List[Tuple[float, float]]] = {}

    for spk in speakers_all:
        pure_segs = pure_regions_by_speaker.get(spk, [])
        overlap_segs = overlap_regions_by_speaker.get(spk, [])
        speech_segs = speech_regions_by_speaker.get(spk, [])

        valid_enroll = pick_valid_or_backup_from_regions(
            pure_segs,
            min_len=min_enroll_len,
            max_len=max_enroll_len,
            max_num=max_enroll_per_speaker_global,
        )
        backup_enroll = pick_valid_or_backup_from_regions(
            overlap_segs,
            min_len=min_enroll_len,
            max_len=max_enroll_len,
            max_num=max_enroll_per_speaker_global,
        )
        invalid_enroll = pick_invalid_from_speech_regions(
            speech_segs,
            max_len=max_enroll_len,
            max_num=max_enroll_per_speaker_global,
        )

        global_valid_enroll_by_speaker[spk] = valid_enroll
        global_backup_enroll_by_speaker[spk] = backup_enroll
        global_invalid_enroll_by_speaker[spk] = invalid_enroll

    # 6) Chunking

    chunks = build_chunks_from_silence_avoid_speech(
        audio_duration=audio_duration,
        silence_intervals=silence_intervals,
        raw_segments_all=raw_segments,
        min_chunk_len=min_chunk_len,
        max_chunk_len=max_chunk_len,
        guard_eps=0.02,
    )

    # 7) Per-chunk assignment
    chunks_info = []

    for (c_start, c_end) in chunks:
        chunk_dict = {
            "start": float(c_start),
            "end": float(c_end),
            "speakers": {}
        }

        for spk in speakers_all:
            # IMPORTANT CHANGE:
            # diarization_segments are exported from RAW segments (unmerged) for better timestamp precision
            raw_segs_spk = raw_segments_by_speaker.get(spk, [])
            diarization_segments = get_diarization_segments_in_chunk_for_speaker(
                c_start,
                c_end,
                raw_segs_spk,      # <-- RAW, NOT merged
                min_overlap=0.01,  # smaller overlap threshold to keep fine segments
            )

            # For enroll selection logic we still use merged speech/pure/overlap
            merged_speech_spk = speech_regions_by_speaker.get(spk, [])
            pure_segs_spk = pure_regions_by_speaker.get(spk, [])
            overlap_segs_spk = overlap_regions_by_speaker.get(spk, [])

            # collect candidates in priority order
            valid_in_chunk = pick_valid_or_backup_in_chunk(
                c_start, c_end, pure_segs_spk,
                min_len=min_enroll_len,
                max_len=max_enroll_len,
                max_num=max_enroll_per_speaker_per_chunk,
            )
            global_valid = global_valid_enroll_by_speaker.get(spk, [])
            backup_in_chunk = pick_valid_or_backup_in_chunk(
                c_start, c_end, overlap_segs_spk,
                min_len=min_enroll_len,
                max_len=max_enroll_len,
                max_num=max_enroll_per_speaker_per_chunk,
            )
            global_backup = global_backup_enroll_by_speaker.get(spk, [])
            invalid_in_chunk = pick_invalid_in_chunk_from_speech(
                c_start, c_end, merged_speech_spk,
                max_len=max_enroll_len,
                max_num=max_enroll_per_speaker_per_chunk,
            )
            global_invalid = global_invalid_enroll_by_speaker.get(spk, [])

            candidates: List[Dict[str, float]] = []

            def add_from_list(src: List[Tuple[float, float]], tag: str):
                nonlocal candidates
                for (s, e) in src:
                    if len(candidates) >= max_enroll_per_speaker_per_chunk:
                        break
                    candidates.append({"start": float(s), "end": float(e), "type": tag})

            # Priority:
            add_from_list(valid_in_chunk, "valid")
            if len(candidates) < max_enroll_per_speaker_per_chunk:
                add_from_list(global_valid, "valid")
            if len(candidates) < max_enroll_per_speaker_per_chunk:
                add_from_list(backup_in_chunk, "backup")
            if len(candidates) < max_enroll_per_speaker_per_chunk:
                add_from_list(global_backup, "backup")
            if len(candidates) < max_enroll_per_speaker_per_chunk:
                add_from_list(invalid_in_chunk, "invalid")
            if len(candidates) < max_enroll_per_speaker_per_chunk:
                add_from_list(global_invalid, "invalid")

            # If still fewer than max_enroll_per_speaker_per_chunk, repeat them cyclically
            if candidates:
                base = list(candidates)
                i = 0
                while len(candidates) < max_enroll_per_speaker_per_chunk:
                    seg = base[i % len(base)]
                    candidates.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "type": seg["type"],
                    })
                    i += 1
            else:
                candidates = []

            chunk_dict["speakers"][spk] = {
                "enroll_segments": candidates,
                "diarization_segments": diarization_segments,  # RAW
            }

        chunks_info.append(chunk_dict)

    # 8) Build global_enroll summary with type tags
    global_enroll: Dict[str, List[Dict[str, float]]] = {}

    for spk in speakers_all:
        entries: List[Dict[str, float]] = []
        for (s, e) in global_valid_enroll_by_speaker.get(spk, []):
            entries.append({"start": float(s), "end": float(e), "type": "valid"})
        for (s, e) in global_backup_enroll_by_speaker.get(spk, []):
            entries.append({"start": float(s), "end": float(e), "type": "backup"})
        for (s, e) in global_invalid_enroll_by_speaker.get(spk, []):
            entries.append({"start": float(s), "end": float(e), "type": "invalid"})
        global_enroll[spk] = entries

    result = {
        "audio_path": audio_path,
        "chunks": chunks_info,
        "global_enroll": global_enroll,
    }
    return result


# =========================
# 8. CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Diarization-based chunking and enroll selection"
    )
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token. If omitted, use HF_TOKEN env var.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to output JSON file. If omitted, print to stdout.")

    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers (optional hint to the diarization model).")

    parser.add_argument("--min_chunk_len", type=float, default=5.0, help="Minimum chunk length in seconds.")
    parser.add_argument("--max_chunk_len", type=float, default=10.0, help="Maximum chunk length in seconds.")

    parser.add_argument("--min_enroll_len", type=float, default=1.5, help="Minimum valid/backup enroll length in seconds.")
    parser.add_argument("--max_enroll_len", type=float, default=4.0, help="Maximum enroll segment length in seconds.")

    parser.add_argument("--max_enroll_per_speaker_global", type=int, default=10, help="Max global enroll segments per speaker.")
    parser.add_argument("--max_enroll_per_speaker_per_chunk", type=int, default=8,
                        help="Enroll segments per speaker per chunk (will repeat if not enough).")

    parser.add_argument("--min_silence_len", type=float, default=0.5, help="Minimum silence length (seconds) used to cut chunks.")

    args = parser.parse_args()

    result = process_audio_with_diarization(
        audio_path=args.audio_path,
        hf_token=args.hf_token,
        num_speakers=args.num_speakers,
        min_chunk_len=args.min_chunk_len,
        max_chunk_len=args.max_chunk_len,
        min_enroll_len=args.min_enroll_len,
        max_enroll_len=args.max_enroll_len,
        max_enroll_per_speaker_global=args.max_enroll_per_speaker_global,
        max_enroll_per_speaker_per_chunk=args.max_enroll_per_speaker_per_chunk,
        min_silence_len=args.min_silence_len,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON to {args.output_json}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
