#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @ hwang258@jhu.edu
#
# SoloSpeech TSE inference with:
#   - chunk/enroll/diarization info from JSON (produced by your diarization+chunking script)
#   - multi-enroll candidates per speaker per chunk
#   - +1 seed per enroll candidate to avoid identical generations
#   - extra diarization-masked enroll candidate (mixture masked by diar spans)
#   - VAD using pydub.detect_nonsilent (rel_to_max or abs)
#   - reranking using: score = recall - alpha * fp_rate
#       recall: predicted non-silent overlaps target speaker diarization
#       fp_rate: predicted non-silent outside target diarization, normalized by non-target duration
#
# Also saves:
#   - ALL candidate wavs for debug
#   - best chunk-level wav per speaker
#   - merged full-length wav per speaker (concatenated chunks)

import os
import json
import yaml
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler

from solospeech.model.solospeech.conditioners import SoloSpeech_TSE
from solospeech.scripts.solospeech.utils import save_audio
from solospeech.vae_modules.autoencoder_wrapper import Autoencoder

try:
    import pydub
except ImportError:
    pydub = None


# =========================
# Reproducibility
# =========================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed set to {seed}")


# =========================
# Diffusion sampling
# =========================

@torch.no_grad()
def sample_diffusion(
    tse_model,
    autoencoder,
    scheduler,
    device,
    mixture: torch.Tensor,
    reference: torch.Tensor,
    lengths: torch.Tensor,
    reference_lengths: torch.Tensor,
    ddim_steps: int = 50,
    eta: float = 0.0,
    seed: int = 2025,
) -> torch.Tensor:
    """
    Run DDIM sampling for SoloSpeech TSE.

    Args:
        mixture:   [B, T_lat, C]
        reference: [B, T_ref_lat, C]
        lengths: [B]
        reference_lengths: [B]
    Returns:
        wav: [B, T_wav]
    """
    generator = torch.Generator(device=device).manual_seed(int(seed))
    scheduler.set_timesteps(ddim_steps)

    # same shape as mixture latent
    x = torch.randn(mixture.shape, generator=generator, device=device)

    for t in scheduler.timesteps:
        x = scheduler.scale_model_input(x, t)
        model_output, _ = tse_model(
            x=x,
            timesteps=t,
            mixture=mixture,
            reference=reference,
            x_len=lengths,
            ref_len=reference_lengths,
        )
        x = scheduler.step(
            model_output=model_output,
            timestep=t,
            sample=x,
            eta=eta,
            generator=generator,
        ).prev_sample

    # Decode latent to waveform via autoencoder
    wav = autoencoder(embedding=x.transpose(2, 1)).squeeze(1)  # [B, T]
    return wav


# =========================
# Output directory helper
# =========================

def ensure_output_dir(output_path: str) -> str:
    """
    Interpret output_path as a directory or derive a directory from it.
    If output_path looks like a file path (*.wav, *.flac, etc.), use its dirname.
    Otherwise, treat it as a directory.
    """
    audio_exts = (".wav", ".flac", ".mp3", ".ogg")
    if output_path.lower().endswith(audio_exts):
        out_dir = os.path.dirname(output_path) or "."
    else:
        out_dir = output_path
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# =========================
# Enroll helpers
# =========================

def pad_enroll_if_needed(enroll_wav: np.ndarray, sr: int, min_enroll_sec: float) -> np.ndarray:
    """Zero-pad enroll_wav to at least min_enroll_sec (avoids encoder conv kernel > length error)."""
    min_samples = int(round(min_enroll_sec * sr))
    if enroll_wav.shape[0] < min_samples:
        enroll_wav = np.pad(enroll_wav, (0, min_samples - enroll_wav.shape[0]), mode="constant")
    return enroll_wav


def load_segment_from_mixture(
    mixture: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> Optional[np.ndarray]:
    """Slice [start_sec, end_sec] from mixture. Return None if empty."""
    s = int(round(float(start_sec) * sr))
    e = int(round(float(end_sec) * sr))
    s = max(0, s)
    e = min(len(mixture), e)
    if e <= s:
        return None
    return mixture[s:e]


def build_diarization_masked_enroll_for_chunk(
    mixture_chunk: np.ndarray,
    diar_segments_global: List[List[float]],
    chunk_start: float,
    sr: int,
    min_enroll_sec: float,
) -> Optional[np.ndarray]:
    """
    Extra enroll candidate based on diarization:
      - Take mixture chunk
      - Keep only times where target speaker is active (per diarization)
      - Set other times to 0
    """
    if mixture_chunk is None or mixture_chunk.shape[0] == 0:
        return None
    if diar_segments_global is None or len(diar_segments_global) == 0:
        return None

    n = mixture_chunk.shape[0]
    times_global = chunk_start + (np.arange(n) / float(sr))
    mask = np.zeros(n, dtype=bool)
    for seg in diar_segments_global:
        if len(seg) != 2:
            continue
        s, e = float(seg[0]), float(seg[1])
        mask |= (times_global >= s) & (times_global <= e)

    if not mask.any():
        return None

    diar_enroll = mixture_chunk.copy()
    diar_enroll[~mask] = 0.0
    diar_enroll = pad_enroll_if_needed(diar_enroll, sr, min_enroll_sec)
    return diar_enroll


# =========================
# Meta-style VAD via pydub.detect_nonsilent
# =========================

def _require_pydub():
    if pydub is None:
        raise ImportError(
            "pydub is not installed. Please run: pip install pydub\n"
            "Also ensure ffmpeg/avlib is available on the system for some formats."
        )


def wav_to_pydub_audiosegment(wav: np.ndarray, sr: int) -> "pydub.AudioSegment":
    """
    Convert float waveform [-1,1] to pydub.AudioSegment (int16 PCM).
    """
    _require_pydub()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        wav = np.zeros((1,), dtype=np.float32)

    # IMPORTANT: clip into [-1, 1] for PCM scaling
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)

    return pydub.AudioSegment(
        data=pcm16.tobytes(),
        sample_width=2,
        frame_rate=int(sr),
        channels=1,
    )


def get_peak_rms(audio: "pydub.AudioSegment", win_ms: int = 250, hop_ms: int = 100) -> float:
    """
    Compute peak RMS ratio over sliding windows (normalized by max_possible_amplitude).
    """
    last_slice_start = len(audio) - win_ms
    if last_slice_start < 0:
        return max(audio.rms / audio.max_possible_amplitude, 0.0)

    peak_rms = -1.0
    for i in range(0, last_slice_start + 1, hop_ms):
        audio_slice = audio[i: i + win_ms]
        peak_rms = max(peak_rms, audio_slice.rms / audio.max_possible_amplitude)
    return max(peak_rms, 0.0)


def detect_nonsilent_from_wav(
    wav: np.ndarray,
    sr: int,
    min_sil_ms: int = 250,
    sil_threshold_db: float = -40.0,
    threshold_mode: str = "rel_to_max",  # {"abs","rel_to_max"}
    target_sr: int = 24000,
    seek_step: int = 10,
    peak_win_ms: int = 250,
    peak_hop_ms: int = 100,
    normalize_peak: bool = True,
) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Meta-style VAD:
      - optionally peak-normalize waveform per candidate to reduce gain drift
      - convert to pydub AudioSegment, resample to target_sr
      - if rel_to_max, convert sil_threshold_db into absolute threshold via peak RMS
      - run pydub.silence.detect_nonsilent
    Returns:
      - spans in seconds, local to this wav (0..dur)
      - debug dict {sil_threshold_db_abs, peak_rms, ...}
    """
    _require_pydub()

    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return [], {"sil_threshold_db_abs": float(sil_threshold_db), "peak_rms": 0.0, "note": "empty_wav"}

    if normalize_peak:
        peak = float(np.max(np.abs(wav)) + 1e-8)
        wav = wav / peak

    audio = wav_to_pydub_audiosegment(wav, sr)
    audio = audio.set_frame_rate(int(target_sr))

    dbg = {"threshold_mode": threshold_mode, "sil_threshold_db_input": float(sil_threshold_db)}

    if threshold_mode == "rel_to_max":
        peak_rms = get_peak_rms(audio, win_ms=peak_win_ms, hop_ms=peak_hop_ms)
        # convert peak_rms ratio -> dB and add to sil_threshold_db => absolute db threshold
        sil_threshold_db_abs = float(sil_threshold_db + pydub.utils.ratio_to_db(peak_rms))
        dbg["peak_rms"] = float(peak_rms)
        # near-silent: skip ratio_to_db to avoid -inf and broken detect_nonsilent
        if peak_rms < 1e-6:
            dbg["sil_threshold_db_abs"] = None
            dbg["note"] = "near_silent_skip"
            return [], dbg
        dbg["sil_threshold_db_abs"] = float(sil_threshold_db_abs)
    elif threshold_mode == "abs":
        sil_threshold_db_abs = float(sil_threshold_db)
        dbg["peak_rms"] = None
        dbg["sil_threshold_db_abs"] = float(sil_threshold_db_abs)
    else:
        raise ValueError(f"Unknown threshold_mode={threshold_mode} (expect 'abs' or 'rel_to_max')")

    spans_ms = pydub.silence.detect_nonsilent(
        audio,
        min_silence_len=int(min_sil_ms),
        silence_thresh=float(sil_threshold_db_abs),
        seek_step=int(seek_step),
    )

    spans = [(float(s) / 1000.0, float(e) / 1000.0) for (s, e) in spans_ms]
    return spans, dbg


# =========================
# Span metrics: recall & fp_rate
# =========================

def _span_len(s: Tuple[float, float]) -> float:
    return max(0.0, float(s[1]) - float(s[0]))


def _inter_len(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def spans_total_length(spans: List[Tuple[float, float]]) -> float:
    return sum(_span_len(s) for s in spans)


def spans_total_intersection(hyp: List[Tuple[float, float]], ref: List[Tuple[float, float]]) -> float:
    inter = 0.0
    for h in hyp:
        for r in ref:
            inter += _inter_len(h, r)
    return inter


def compute_recall_and_fp_rate(
    hyp_spans_local: List[Tuple[float, float]],
    ref_spans_global: List[List[float]],
    chunk_start: float,
    chunk_end: float,
) -> Dict[str, float]:
    """
    hyp_spans_local: predicted non-silent spans (seconds), local to chunk audio [0..chunk_dur]
    ref_spans_global: diarization spans (global seconds) for target speaker in this chunk
    """
    chunk_dur = float(chunk_end - chunk_start)
    if chunk_dur <= 1e-8:
        return {"recall": 0.0, "fp_rate": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "ref_len": 0.0, "hyp_len": 0.0}

    # clip hyp to [0, chunk_dur]
    hyp = []
    for s, e in hyp_spans_local:
        s = max(0.0, float(s))
        e = min(chunk_dur, float(e))
        if e > s:
            hyp.append((s, e))

    # convert ref global -> local and clip
    ref = []
    for seg in (ref_spans_global or []):
        if len(seg) != 2:
            continue
        s, e = float(seg[0]), float(seg[1])
        s = max(s, float(chunk_start))
        e = min(e, float(chunk_end))
        if e > s:
            ref.append((s - float(chunk_start), e - float(chunk_start)))

    hyp_len = spans_total_length(hyp)
    ref_len = spans_total_length(ref)
    tp = spans_total_intersection(hyp, ref)
    fp = max(0.0, hyp_len - tp)
    fn = max(0.0, ref_len - tp)

    recall = tp / ref_len if ref_len > 1e-8 else 0.0
    not_ref_len = max(1e-8, chunk_dur - ref_len)
    fp_rate = fp / not_ref_len

    return {
        "recall": float(recall),
        "fp_rate": float(fp_rate),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "ref_len": float(ref_len),
        "hyp_len": float(hyp_len),
    }


# =========================
# Main
# =========================

def main(args):
    set_seed(args.random_seed)
    out_dir = ensure_output_dir(args.output_path)

    # Load JSON meta (chunks / speakers / enroll / diarization)
    with open(args.json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = meta.get("chunks", [])
    if not chunks:
        raise ValueError(f"No 'chunks' found in JSON: {args.json_path}")

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using device: {device}")

    # Config paths
    tse_config_path = os.path.join(args.local_dir, "config_extractor.yaml")
    vae_config_path = os.path.join(args.local_dir, "config_compressor.json")
    autoencoder_path = os.path.join(args.local_dir, "compressor.ckpt")
    tse_ckpt_path = os.path.join(args.local_dir, "extractor.pt")

    # Load TSE config
    print("[Model] Loading configs and models...")
    with open(tse_config_path, "r") as fp:
        tse_config = yaml.safe_load(fp)

    # Compressor
    autoencoder = Autoencoder(autoencoder_path, vae_config_path, "stable_vae", quantization_first=True)
    autoencoder.eval()
    autoencoder.to(device)

    # Extractor
    tse_model = SoloSpeech_TSE(
        tse_config["diffwrap"]["UDiT"],
        tse_config["diffwrap"]["ViT"],
    ).to(device)
    tse_model.load_state_dict(torch.load(tse_ckpt_path, map_location=device)["model"])
    tse_model.eval()

    # Scheduler
    noise_scheduler = DDIMScheduler(**tse_config["ddim"]["diffusers"])

    # Touch scheduler once to set dtypes
    latents = torch.randn((1, 128, 128), device=device)
    noise = torch.randn(latents.shape, device=device)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=latents.device
    ).long()
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    # Load mixture once
    print("[Audio] Loading mixture audio...")
    mixture, sr = librosa.load(args.test_wav, sr=args.sample_rate)
    print(f"[Audio] Mixture length: {len(mixture)/sr:.2f}s, sr={sr}")

    speaker_outputs: Dict[str, List[np.ndarray]] = {}  # for merged final wav

    # Optional: dump VAD debug JSONL
    vad_debug_path = os.path.join(out_dir, "vad_debug.jsonl") if args.dump_vad_debug else None
    if vad_debug_path:
        print(f"[Debug] Will write VAD debug to: {vad_debug_path}")

    print("[Run] Start chunk-wise extraction...")

    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Chunks")):
        c_start = float(chunk["start"])
        c_end = float(chunk["end"])
        if c_end <= c_start:
            continue

        # Mixture chunk
        s_samp = int(round(c_start * sr))
        e_samp = int(round(c_end * sr))
        s_samp = max(0, s_samp)
        e_samp = min(len(mixture), e_samp)
        if e_samp <= s_samp:
            continue

        mixture_chunk = mixture[s_samp:e_samp]

        speakers = chunk.get("speakers", {})
        if not speakers:
            continue

        # Pre-encode mixture chunk once (saves time)
        mixture_tensor = torch.tensor(mixture_chunk, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T]
        with torch.no_grad():
            mix_lat = autoencoder(audio=mixture_tensor.unsqueeze(1))  # [1,C,T_lat]
        mix_len = mix_lat.shape[-1]
        lengths = torch.LongTensor([mix_len]).to(device)
        mixture_cond = mix_lat.transpose(2, 1)  # [1, T_lat, C]

        for spk_id, spk_info in speakers.items():
            enroll_segments = spk_info.get("enroll_segments", [])
            diar_segments = spk_info.get("diarization_segments", [])

            if not enroll_segments:
                continue

            cand_wavs: List[Optional[np.ndarray]] = []
            cand_scores: List[float] = []
            cand_metrics: List[Dict[str, Any]] = []
            cand_tags: List[str] = []

            # ---------- JSON enroll candidates ----------
            for k in range(args.num_candidates):
                enroll = enroll_segments[k % len(enroll_segments)]
                if isinstance(enroll, dict):
                    e_start = float(enroll["start"])
                    e_end = float(enroll["end"])
                    e_type = str(enroll.get("type", "unknown"))
                else:
                    e_start = float(enroll[0])
                    e_end = float(enroll[1])
                    e_type = "unknown"

                enroll_wav = load_segment_from_mixture(mixture, sr, e_start, e_end)
                if enroll_wav is None or enroll_wav.size == 0:
                    cand_wavs.append(None)
                    cand_scores.append(-1e9)
                    cand_metrics.append({"note": "empty_enroll"})
                    cand_tags.append(e_type)
                    continue

                enroll_wav = pad_enroll_if_needed(enroll_wav, sr, args.min_enroll_sec)
                # if int(args.min_enroll_sec * sr) > enroll_wav.shape[0]:
                #     continue

                enroll_tensor = torch.tensor(enroll_wav, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    ref_lat = autoencoder(audio=enroll_tensor.unsqueeze(1))  # [1,C,T_ref]
                ref_len = ref_lat.shape[-1]
                ref_lengths = torch.LongTensor([ref_len]).to(device)
                enroll_cond = ref_lat.transpose(2, 1)  # [1,T_ref,C]

                seed = int(args.random_seed + k)
                with torch.no_grad():
                    pred_t = sample_diffusion(
                        tse_model=tse_model,
                        autoencoder=autoencoder,
                        scheduler=noise_scheduler,
                        device=device,
                        mixture=mixture_cond,
                        reference=enroll_cond,
                        lengths=lengths,
                        reference_lengths=ref_lengths,
                        ddim_steps=args.num_infer_steps,
                        eta=args.eta,
                        seed=seed,
                    )
                pred_wav = pred_t[0].detach().cpu().numpy()

                # VAD spans (local)
                hyp_spans, vad_dbg = detect_nonsilent_from_wav(
                    wav=pred_wav,
                    sr=sr,
                    min_sil_ms=args.vad_min_sil_ms,
                    sil_threshold_db=args.vad_sil_threshold_db,
                    threshold_mode=args.vad_threshold_mode,
                    target_sr=args.vad_target_sr,
                    seek_step=args.vad_seek_step,
                    peak_win_ms=args.vad_peak_win_ms,
                    peak_hop_ms=args.vad_peak_hop_ms,
                    normalize_peak=bool(args.vad_normalize_peak),
                )

                metrics = compute_recall_and_fp_rate(
                    hyp_spans_local=hyp_spans,
                    ref_spans_global=diar_segments,
                    chunk_start=c_start,
                    chunk_end=c_end,
                )
                score = float(metrics["recall"] - args.alpha_fp * metrics["fp_rate"])

                # Debug prints
                if args.print_vad:
                    print(
                        f"[VAD] chunk={chunk_idx} spk={spk_id} cand={k} "
                        f"thr_db_abs={vad_dbg.get('sil_threshold_db_abs', None)} "
                        f"hyp_spans={len(hyp_spans)}"
                    )
                    print(f"[VAD intervals] {hyp_spans}")

                print(
                    f"[Score] Chunk {chunk_idx}, spk {spk_id}, cand {k}: "
                    f"recall={metrics['recall']:.3f}, fp_rate={metrics['fp_rate']:.3f}, "
                    f"score={score:.3f} (enroll_type={e_type})"
                )

                if vad_debug_path:
                    rec = {
                        "chunk": chunk_idx,
                        "speaker": spk_id,
                        "cand": k,
                        "enroll_type": e_type,
                        "chunk_start": c_start,
                        "chunk_end": c_end,
                        "vad_hyp_spans": hyp_spans,
                        "diar_ref_spans": diar_segments,
                        "vad_dbg": vad_dbg,
                        "metrics": metrics,
                        "score": score,
                        "seed": seed,
                        "enroll_start": e_start,
                        "enroll_end": e_end,
                    }
                    with open(vad_debug_path, "a", encoding="utf-8") as wf:
                        wf.write(json.dumps(rec) + "\n")

                cand_wavs.append(pred_wav)
                cand_scores.append(score)
                cand_metrics.append({"vad_dbg": vad_dbg, "hyp_spans": hyp_spans, **metrics})
                cand_tags.append(e_type)

            # ---------- Extra diarization-masked enroll candidate ----------
            if args.add_diar_mask_enroll:
                diar_enroll = build_diarization_masked_enroll_for_chunk(
                    mixture_chunk=mixture_chunk,
                    diar_segments_global=diar_segments,
                    chunk_start=c_start,
                    sr=sr,
                    min_enroll_sec=args.min_enroll_sec,
                )
                if diar_enroll is not None:
                    enroll_tensor = torch.tensor(diar_enroll, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        ref_lat = autoencoder(audio=enroll_tensor.unsqueeze(1))
                    ref_len = ref_lat.shape[-1]
                    ref_lengths = torch.LongTensor([ref_len]).to(device)
                    enroll_cond = ref_lat.transpose(2, 1)

                    seed = int(args.random_seed + args.num_candidates)  # fixed extra seed
                    with torch.no_grad():
                        pred_t = sample_diffusion(
                            tse_model=tse_model,
                            autoencoder=autoencoder,
                            scheduler=noise_scheduler,
                            device=device,
                            mixture=mixture_cond,
                            reference=enroll_cond,
                            lengths=lengths,
                            reference_lengths=ref_lengths,
                            ddim_steps=args.num_infer_steps,
                            eta=args.eta,
                            seed=seed,
                        )
                    pred_wav = pred_t[0].detach().cpu().numpy()

                    hyp_spans, vad_dbg = detect_nonsilent_from_wav(
                        wav=pred_wav,
                        sr=sr,
                        min_sil_ms=args.vad_min_sil_ms,
                        sil_threshold_db=args.vad_sil_threshold_db,
                        threshold_mode=args.vad_threshold_mode,
                        target_sr=args.vad_target_sr,
                        seek_step=args.vad_seek_step,
                        peak_win_ms=args.vad_peak_win_ms,
                        peak_hop_ms=args.vad_peak_hop_ms,
                        normalize_peak=bool(args.vad_normalize_peak),
                    )
                    metrics = compute_recall_and_fp_rate(
                        hyp_spans_local=hyp_spans,
                        ref_spans_global=diar_segments,
                        chunk_start=c_start,
                        chunk_end=c_end,
                    )
                    score = float(metrics["recall"] - args.alpha_fp * metrics["fp_rate"])

                    cand_wavs.append(pred_wav)
                    cand_scores.append(score)
                    cand_metrics.append({"vad_dbg": vad_dbg, "hyp_spans": hyp_spans, **metrics})
                    cand_tags.append("diar_mask")

                    kk = len(cand_wavs) - 1
                    print(
                        f"[Score] Chunk {chunk_idx}, spk {spk_id}, cand {kk}: "
                        f"recall={metrics['recall']:.3f}, fp_rate={metrics['fp_rate']:.3f}, "
                        f"score={score:.3f} (enroll_type=diar_mask)"
                    )
                else:
                    print(f"[Info] Chunk {chunk_idx}, spk {spk_id}: no diar_mask enroll available.")

            # ---------- Save ALL candidates ----------
            if args.save_all_candidates:
                for kk, w in enumerate(cand_wavs):
                    if w is None:
                        continue
                    out_all = os.path.join(
                        out_dir,
                        f"chunk{chunk_idx:04d}_{str(spk_id).replace(' ', '_')}_candidate{kk}.wav"
                    )
                    save_audio(out_all, args.sample_rate, torch.tensor(w, dtype=torch.float32).unsqueeze(0))
                    print(f"[SaveALL] Chunk {chunk_idx}, speaker {spk_id}, candidate {kk}: {out_all}")

            # ---------- Pick best ----------
            if not cand_scores:
                continue
            best_idx = int(np.argmax(np.array(cand_scores)))
            best_wav = cand_wavs[best_idx]
            if best_wav is None or best_wav.size == 0:
                continue

            spk_clean = str(spk_id).replace(" ", "_")
            out_best = os.path.join(out_dir, f"chunk{chunk_idx:04d}_{spk_clean}.wav")
            save_audio(out_best, args.sample_rate, torch.tensor(best_wav, dtype=torch.float32).unsqueeze(0))

            print(
                f"[Save] Chunk {chunk_idx}, speaker {spk_id}, best_cand={best_idx}, "
                f"score={cand_scores[best_idx]:.3f}, path={out_best}"
            )

            # Accumulate for speaker merge
            speaker_outputs.setdefault(spk_id, []).append(best_wav)

    # ---------- Merge all chunk outputs per speaker ----------
    if args.save_merged_speakers:
        print("[Final] Merging chunk outputs per speaker...")
        for spk_id, wav_list in speaker_outputs.items():
            if not wav_list:
                continue
            merged = np.concatenate(wav_list, axis=0)
            out_path = os.path.join(out_dir, f"speaker_{str(spk_id).replace(' ', '_')}.wav")
            save_audio(out_path, args.sample_rate, torch.tensor(merged, dtype=torch.float32).unsqueeze(0))
            print(f"[Final Save] Speaker {spk_id}: {out_path}, duration={len(merged)/args.sample_rate:.2f}s")

    print("[Done] All chunks processed.")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SoloSpeech TSE inference with pydub VAD spans reranking (Meta-style)."
    )
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output directory or a file path (directory will be derived).")
    parser.add_argument("--test-wav", type=str, required=True,
                        help="Path to the input mixture wav.")
    parser.add_argument("--json-path", type=str, required=True,
                        help="Path to the JSON file with chunks/enroll/diarization info.")
    parser.add_argument("--local-dir", type=str, required=True,
                        help="Directory containing config_extractor.yaml, config_compressor.json, compressor.ckpt, extractor.pt.")

    # Diffusion
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter.")
    parser.add_argument("--num_infer_steps", type=int, default=200, help="Number of DDIM steps.")
    parser.add_argument("--num_candidates", type=int, default=8,
                        help="Number of enroll candidates per chunk per speaker from JSON.")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate for loading and saving audio.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Base random seed. Each candidate uses seed+idx.")

    # Enroll safety
    parser.add_argument("--min_enroll_sec", type=float, default=0.8,
                        help="Minimum enroll duration (sec) after padding to avoid conv kernel error.")
    parser.add_argument("--add_diar_mask_enroll", action="store_true",
                        help="Add extra diarization-masked enroll candidate (type=diar_mask).")

    # VAD
    parser.add_argument("--vad_min_sil_ms", type=int, default=200,
                        help="pydub.detect_nonsilent: min_silence_len in ms.")
    parser.add_argument("--vad_sil_threshold_db", type=float, default=-40.0,
                        help="Base silence threshold in dBFS (Meta snippet uses -40).")
    parser.add_argument("--vad_threshold_mode", type=str, default="rel_to_max",
                        choices=["abs", "rel_to_max"],
                        help="Threshold mode: 'rel_to_max' or 'abs'.")
    parser.add_argument("--vad_target_sr", type=int, default=16000,
                        help="Resample audio to this SR before VAD.")
    parser.add_argument("--vad_seek_step", type=int, default=10,
                        help="pydub.detect_nonsilent seek_step (ms).")
    parser.add_argument("--vad_peak_win_ms", type=int, default=200,
                        help="Peak RMS window length in ms for rel_to_max.")
    parser.add_argument("--vad_peak_hop_ms", type=int, default=100,
                        help="Peak RMS hop length in ms for rel_to_max.")
    parser.add_argument("--vad_normalize_peak", type=int, default=1,
                        help="1: peak-normalize each candidate before VAD (recommended). 0: disable.")
    parser.add_argument("--print_vad", action="store_true",
                        help="Print VAD spans for each candidate.")

    # Reranking
    parser.add_argument("--alpha_fp", type=float, default=1.0,
                        help="Penalty weight for fp_rate: score = recall - alpha_fp * fp_rate")

    # Saving & debug
    parser.add_argument("--save_all_candidates", action="store_true",
                        help="Save all candidates wavs for debug.")
    parser.add_argument("--save_merged_speakers", action="store_true",
                        help="Save merged full-length wav for each speaker.")
    parser.add_argument("--dump_vad_debug", action="store_true",
                        help="Write a vad_debug.jsonl containing spans/metrics per candidate.")

    args = parser.parse_args()
    main(args)
