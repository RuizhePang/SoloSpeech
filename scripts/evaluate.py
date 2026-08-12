#!/usr/bin/env python3

import csv
import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import hydra
import librosa
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf


METRIC_GROUPS = {
    "compressor": ["compressor_sisdr", "compressor_pesq", "compressor_estoi"],
    "extractor": ["extractor_sisdr", "extractor_pesq", "extractor_estoi"],
    "corrector": ["corrector_sisdr", "corrector_pesq", "corrector_estoi"],
    "system": ["system_sisdr", "system_dnsmos", "system_wer"],
}
METRIC_GROUPS["all"] = [
    metric
    for group_name in ("compressor", "extractor", "corrector", "system")
    for metric in METRIC_GROUPS[group_name]
]


def hydra_choice(name):
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.choices[name]


def get_test_data(cfg):
    value = cfg.evaluation.get("test_name", None)
    if is_null_value(value):
        value = cfg.evaluation.get("test_data", "Libri2Mix")
    return str(value)


def get_test_data_root(cfg):
    test_dir = cfg.evaluation.get("test_dir", None)
    if is_null_value(test_dir):
        test_dir = cfg.evaluation.get("test_data_dir", None)
    if not is_null_value(test_dir):
        return Path(test_dir)
    return Path(cfg.data_dir) / get_test_data(cfg)


def get_librimix_storage_root(cfg):
    return get_test_data_root(cfg) / "LibriMixData"


def get_raw_dataset_name(cfg):
    raw_dataset_name = cfg.evaluation.get("raw_dataset_name", None)
    return str(raw_dataset_name) if not is_null_value(raw_dataset_name) else get_test_data(cfg)


def get_evaluation_root(cfg):
    root = Path(cfg.save_dir) / "evaluation"
    test_data = get_test_data(cfg)
    return root if test_data == "Libri2Mix" else root / test_data


def is_null_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "none", "null")
    return False


def get_manifest_path(cfg):
    manifest = cfg.evaluation.get("manifest", None)
    if is_null_value(manifest):
        return None
    return Path(str(manifest)).expanduser()


def normalize_manifest_row(row):
    return {str(key).strip().lower(): value for key, value in row.items() if key is not None}


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("items") or data.get("data") or data.get("utterances")
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must be a list or contain items/data/utterances: {path}")
        return [normalize_manifest_row(row) for row in data]

    if suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(normalize_manifest_row(json.loads(line)))
        return rows

    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(newline="") as f:
        return [normalize_manifest_row(row) for row in csv.DictReader(f, delimiter=delimiter)]


def value_for(row, *names):
    for name in names:
        value = row.get(name)
        if not is_null_value(value):
            return value
    return None


def path_for(value, cfg, manifest_dir):
    if is_null_value(value):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        get_test_data_root(cfg) / path,
        manifest_dir / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def text_for(row, category, source_idx):
    idx = "" if source_idx is None else str(source_idx)
    return value_for(
        row,
        f"{category}_transcript{idx}",
        f"{category}_text{idx}",
        f"transcript{idx}",
        f"text{idx}",
        f"ref_text{idx}",
        f"reference_text{idx}",
        "transcript",
        "text",
        "ref_text",
        "reference_text",
    )


def source_for(row, category, source_idx, cfg, manifest_dir):
    idx = "" if source_idx is None else str(source_idx)
    value = value_for(
        row,
        f"{category}_source{idx}",
        f"{category}_source_path{idx}",
        f"source{idx}",
        f"source_path{idx}",
        f"target{idx}",
        f"target_path{idx}",
        f"clean{idx}",
        f"clean_path{idx}",
        f"s{idx}" if source_idx is not None else "source",
        f"s{idx}_path" if source_idx is not None else "source_path",
        "source",
        "source_path",
        "target",
        "target_path",
        "clean",
        "clean_path",
    )
    return path_for(value, cfg, manifest_dir)


def reference_for(row, category, source_idx, cfg, manifest_dir):
    idx = "" if source_idx is None else str(source_idx)
    value = value_for(
        row,
        f"{category}_reference{idx}",
        f"{category}_reference_path{idx}",
        f"{category}_enrollment{idx}",
        f"{category}_enrollment_path{idx}",
        f"reference{idx}",
        f"reference_path{idx}",
        f"enrollment{idx}",
        f"enrollment_path{idx}",
        f"aux{idx}",
        f"aux_path{idx}",
        "reference",
        "reference_path",
        "enrollment",
        "enrollment_path",
        "aux",
        "aux_path",
    )
    return path_for(value, cfg, manifest_dir)


def mix_for(row, category, source_idx, cfg, manifest_dir):
    idx = "" if source_idx is None else str(source_idx)
    return path_for(
        value_for(
            row,
            f"{category}_mix{idx}",
            f"{category}_mix_path{idx}",
            f"mix{idx}",
            f"mix_path{idx}",
            f"mixture{idx}",
            f"mixture_path{idx}",
            f"noisy{idx}",
            f"noisy_path{idx}",
            f"{category}_mix",
            f"{category}_mix_path",
            "mix",
            "mix_path",
            "mixture",
            "mixture_path",
            "noisy",
            "noisy_path",
        ),
        cfg,
        manifest_dir,
    )


def manifest_source_indices(row, category):
    keys = set(row)
    indexed = []
    for idx in (1, 2):
        patterns = (
            f"{category}_source{idx}",
            f"{category}_source_path{idx}",
            f"source{idx}",
            f"source_path{idx}",
            f"target{idx}",
            f"target_path{idx}",
            f"clean{idx}",
            f"clean_path{idx}",
            f"s{idx}",
            f"s{idx}_path",
        )
        if any(key in keys and not is_null_value(row.get(key)) for key in patterns):
            indexed.append(idx)
    return indexed or [None]


def category_allowed(row, category):
    row_category = value_for(row, "category", "metric_group", "type")
    if is_null_value(row_category):
        return True
    categories = [item.strip().lower() for item in str(row_category).split(",")]
    return category in categories or "all" in categories


def get_manifest_audio_pairs(cfg, category):
    manifest_path = get_manifest_path(cfg)
    if manifest_path is None:
        return None

    rows = read_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    pairs = []

    for row_index, row in enumerate(rows):
        if not category_allowed(row, category):
            continue

        for source_idx in manifest_source_indices(row, category):
            source_path = source_for(row, category, source_idx, cfg, manifest_dir)
            if source_path is None:
                continue

            item_id = value_for(row, "id", "utt_id", "uid", "key", "name", "mixture_id", "sample_id")
            if item_id is None:
                item_id = source_path.stem
            if source_idx is not None:
                item_id = f"{item_id}_source{source_idx}"

            pair = {
                "id": str(item_id),
                "source": source_path,
            }
            mix_path = mix_for(row, category, source_idx, cfg, manifest_dir)
            reference_path = reference_for(row, category, source_idx, cfg, manifest_dir)
            if mix_path is not None:
                pair["mix"] = mix_path
            if reference_path is None:
                raise ValueError(f"{pair['id']}: manifest row is missing enrollment/reference")
            pair["reference"] = reference_path

            transcript = text_for(row, category, source_idx)
            if transcript is not None:
                pair["transcript"] = str(transcript)

            pairs.append(pair)

    logger.info(f"Loaded {len(pairs)} {category} pairs from manifest {manifest_path}")
    return pairs


def normalize_metrics(metrics):
    if isinstance(metrics, str):
        raw_metrics = [item.strip() for item in metrics.split(",") if item.strip()]
    elif isinstance(metrics, (list, tuple, ListConfig)):
        raw_metrics = []
        for item in metrics:
            raw_metrics.extend(normalize_metrics(item))
    else:
        raise TypeError(f"Unsupported metrics type: {type(metrics)}")

    expanded = []
    for metric in raw_metrics:
        expanded.extend(METRIC_GROUPS.get(metric, [metric]))
    normalized = list(dict.fromkeys(expanded))
    valid_metrics = set(METRIC_GROUPS["all"])
    invalid = [metric for metric in normalized if metric not in valid_metrics]
    if invalid:
        raise ValueError(f"Unsupported evaluation metrics: {invalid}. Valid metrics: {sorted(valid_metrics)}")
    return normalized


def category_for_metric(metric):
    return metric.split("_", 1)[0]


def metric_name(metric):
    return metric.split("_", 1)[1]


def mean_std(values):
    arr = np.array([value for value in values if value is not None and not np.isnan(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def save_audio_outputs(cfg):
    return bool(cfg.evaluation.get("save_audio_outputs", False))


def save_audio_num_samples(cfg):
    return int(cfg.evaluation.get("save_audio_num_samples", 20))


def save_audio_seed(cfg):
    return int(cfg.evaluation.get("save_audio_seed", 2024))


def format_item_metrics(row):
    fields = []
    for key in sorted(row):
        if key in ("id", "source", "estimate"):
            continue
        value = row[key]
        if isinstance(value, float):
            fields.append(f"{key}={value:.4f}")
        else:
            fields.append(f"{key}={value}")
    return ", ".join(fields) if fields else "no metrics"


def load_audio(path, sample_rate):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def tensor_to_audio_array(audio, label=None):
    audio = audio.detach().float().cpu().squeeze().numpy().astype(np.float32)
    if not np.all(np.isfinite(audio)):
        raise RuntimeError(f"{label or 'audio'} contains NaN/Inf")

    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def trim_pair(reference, estimate):
    length = min(len(reference), len(estimate))
    return reference[:length], estimate[:length]


def si_sdr(estimate, reference, eps=1e-8):
    estimate, reference = trim_pair(reference, estimate)
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    scale = np.sum(estimate * reference) / (np.sum(reference ** 2) + eps)
    projection = scale * reference
    noise = estimate - projection
    return float(10 * np.log10((np.sum(projection ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def compute_pesq(reference, estimate, sample_rate):
    from pesq import pesq

    reference, estimate = trim_pair(reference, estimate)
    return float(pesq(sample_rate, reference, estimate, "wb"))


def compute_estoi(reference, estimate, sample_rate):
    from pystoi import stoi

    reference, estimate = trim_pair(reference, estimate)
    return float(stoi(reference, estimate, sample_rate, extended=True))


class DNSMOS:
    def __init__(self, num_threads=1):
        import onnxruntime as ort

        repo_root = Path(__file__).resolve().parent.parent
        model_dir = repo_root / "solospeech" / "metrics" / "DNSMOS"
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(num_threads)
        session_options.inter_op_num_threads = int(num_threads)
        self.primary = ort.InferenceSession(str(model_dir / "sig_bak_ovr.onnx"), sess_options=session_options)
        self.p808 = ort.InferenceSession(str(model_dir / "model_v8.onnx"), sess_options=session_options)

    @staticmethod
    def audio_melspec(audio, sample_rate, n_mels=120, frame_size=320, hop_length=160):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        return ((librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40).T

    @staticmethod
    def polyfit(sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def score_audio(self, audio, sample_rate):
        target_len = int(9.01 * sample_rate)
        while len(audio) < target_len:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / sample_rate) - 9.01) + 1
        hop_len = sample_rate
        sig_values, bak_values, ovr_values, p808_values = [], [], [], []
        for idx in range(max(num_hops, 1)):
            segment = audio[int(idx * hop_len) : int((idx + 9.01) * sample_rate)]
            if len(segment) < target_len:
                continue
            primary_input = {"input_1": np.array(segment, dtype=np.float32)[np.newaxis, :]}
            p808_input = {
                "input_1": np.array(
                    self.audio_melspec(segment[:-160], sample_rate),
                    dtype=np.float32,
                )[np.newaxis, :, :]
            }
            p808_values.append(float(self.p808.run(None, p808_input)[0][0][0]))
            sig_raw, bak_raw, ovr_raw = self.primary.run(None, primary_input)[0][0]
            sig, bak, ovr = self.polyfit(sig_raw, bak_raw, ovr_raw)
            sig_values.append(float(sig))
            bak_values.append(float(bak))
            ovr_values.append(float(ovr))

        return {
            "dnsmos_sig": float(np.mean(sig_values)),
            "dnsmos_bak": float(np.mean(bak_values)),
            "dnsmos_ovrl": float(np.mean(ovr_values)),
            "dnsmos_p808": float(np.mean(p808_values)),
        }

    def __call__(self, wav_path, sample_rate):
        import soundfile as sf

        audio, input_rate = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if input_rate != sample_rate:
            audio = librosa.resample(audio, orig_sr=input_rate, target_sr=sample_rate)
        return self.score_audio(audio.astype(np.float32), sample_rate)


class WERComputer:
    def __init__(self, cfg):
        from jiwer import wer
        from whisper.normalizers import EnglishTextNormalizer
        import whisper

        self.wer_fn = wer
        self.normalizer = EnglishTextNormalizer()
        self.model = whisper.load_model(cfg.evaluation.whisper_model)
        self.sample_rate = int(cfg.evaluation.sample_rate)
        self.transcripts = load_librispeech_transcripts(cfg)

    def transcribe(self, audio_or_path):
        if isinstance(audio_or_path, (str, Path)):
            audio_or_path = load_audio(audio_or_path, self.sample_rate)
        result = self.model.transcribe(audio_or_path)
        return self.normalizer(result["text"].strip())

    def reference_text(self, source_path):
        utt_id = infer_librispeech_utt_id(source_path)
        if utt_id and utt_id in self.transcripts:
            return self.normalizer(self.transcripts[utt_id])
        return self.transcribe(source_path)

    def __call__(self, source_path, estimate_path, reference_text=None):
        reference = self.normalizer(str(reference_text)) if reference_text else self.reference_text(source_path)
        hypothesis = self.transcribe(estimate_path)
        return float(self.wer_fn(reference, hypothesis))


def load_librispeech_transcripts(cfg):
    librispeech_root = get_librimix_storage_root(cfg) / "LibriSpeech"
    transcripts = {}
    for path in librispeech_root.rglob("*.trans.txt"):
        with path.open() as f:
            for line in f:
                utt_id, text = line.strip().split(" ", 1)
                transcripts[utt_id] = text
    if transcripts:
        logger.info(f"Loaded {len(transcripts)} LibriSpeech transcript entries")
    else:
        logger.warning(f"No LibriSpeech transcripts found under {librispeech_root}")
    return transcripts


def infer_librispeech_utt_id(path):
    stem = Path(path).stem
    match = re.search(r"_source([12])(?:hatP|system)?$", stem)
    if match:
        source_idx = int(match.group(1)) - 1
        mixture_stem = stem[: match.start()]
        utts = mixture_stem.split("_")
        if source_idx < len(utts):
            return utts[source_idx]
    candidates = re.findall(r"\d+-\d+-\d+", stem)
    return candidates[0] if candidates else None


def speakerbeam_csv_dir(cfg):
    return (
        Path(cfg.data_dir)
        / "Libri2Mix"
        / "SpeakerBeamData"
        / cfg.extractor.data.sample_rate_dir
        / cfg.extractor.data.mix_mode
        / cfg.evaluation.split
    )


def speakerbeam_mix_csv(cfg, csv_dir):
    task_candidates = []
    task = str(cfg.extractor.data.get("task", "sep_noisy"))
    if task == "sep_noisy":
        task_candidates.append("mix_both")
    elif task == "sep_clean":
        task_candidates.append("mix_clean")
    task_candidates.extend(["mix_both", "mix_clean"])

    for task_name in dict.fromkeys(task_candidates):
        path = csv_dir / f"mixture_{cfg.evaluation.split}_{task_name}.csv"
        if path.is_file():
            return path
    return None


def first_existing_path(row, cfg, csv_dir, *names):
    value = value_for(row, *names)
    return path_for(value, cfg, csv_dir)


def first_enrollment_path(row, cfg, csv_dir):
    for idx in range(1, 32):
        path = first_existing_path(row, cfg, csv_dir, f"enr_path{idx}", f"enrollment_path{idx}", f"reference_path{idx}")
        if path is not None:
            return path
    return first_existing_path(row, cfg, csv_dir, "enr_path", "enrollment_path", "reference_path")


def get_speakerbeam_audio_pairs(cfg, category):
    csv_dir = speakerbeam_csv_dir(cfg)
    enrollment_csv = csv_dir / "mixture2enrollment.csv"
    mix_csv = speakerbeam_mix_csv(cfg, csv_dir)
    if not enrollment_csv.is_file() or mix_csv is None:
        raise FileNotFoundError(
            f"Missing SpeakerBeam metadata for {category}: expected {enrollment_csv} and mixture CSV under {csv_dir}"
        )

    with mix_csv.open(newline="") as f:
        mixture_rows = {
            normalize_manifest_row(row).get("mixture_id"): normalize_manifest_row(row)
            for row in csv.DictReader(f)
        }

    pairs = []
    with enrollment_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            row = normalize_manifest_row(row)
            mix_id = value_for(row, "mixture_id", "mix_id")
            utt_id = value_for(row, "utterance_id", "utt_id")
            if is_null_value(mix_id) or is_null_value(utt_id) or mix_id not in mixture_rows:
                continue

            utts = str(mix_id).split("_")
            if str(utt_id) not in utts:
                logger.warning(f"Skip {mix_id}/{utt_id}: utterance is not part of mixture id")
                continue
            source_idx = utts.index(str(utt_id)) + 1
            mix_row = mixture_rows[str(mix_id)]
            source_path = first_existing_path(mix_row, cfg, csv_dir, f"source_{source_idx}_path", f"s{source_idx}_path")
            mix_path = first_existing_path(mix_row, cfg, csv_dir, "mixture_path", "mix_path")
            reference_path = first_enrollment_path(row, cfg, csv_dir)
            if source_path is None or mix_path is None or reference_path is None:
                logger.warning(f"Skip {mix_id}/source{source_idx}: missing source, mixture, or enrollment path")
                continue

            pairs.append(
                {
                    "id": f"{mix_id}_source{source_idx}",
                    "source": source_path,
                    "mix": mix_path,
                    "reference": reference_path,
                }
            )

    logger.info(f"Loaded {len(pairs)} {category} pairs from SpeakerBeam metadata {csv_dir}")
    return pairs


def get_compressor_audio_pairs(cfg):
    manifest_pairs = get_manifest_audio_pairs(cfg, "compressor")
    if manifest_pairs is not None:
        return manifest_pairs

    sample_rate_dir = cfg.extractor.data.sample_rate_dir
    mix_mode = cfg.extractor.data.mix_mode
    split = cfg.evaluation.split
    storage_root = get_librimix_storage_root(cfg)
    raw_root = storage_root / get_raw_dataset_name(cfg) / sample_rate_dir / mix_mode / split
    pairs = []
    for source in cfg.evaluation.compressor.sources:
        for wav_path in sorted((raw_root / source).glob("*.wav")):
            pairs.append({"id": wav_path.stem, "source": wav_path, "reference": wav_path})
    return pairs


def get_extractor_audio_pairs(cfg):
    manifest_pairs = get_manifest_audio_pairs(cfg, "extractor")
    if manifest_pairs is not None:
        return manifest_pairs
    return get_speakerbeam_audio_pairs(cfg, "extractor")


def get_corrector_audio_pairs(cfg):
    manifest_pairs = get_manifest_audio_pairs(cfg, "corrector")
    if manifest_pairs is not None:
        return manifest_pairs
    return get_extractor_audio_pairs(cfg)


def get_system_audio_pairs(cfg):
    manifest_pairs = get_manifest_audio_pairs(cfg, "system")
    if manifest_pairs is not None:
        return manifest_pairs
    return get_extractor_audio_pairs(cfg)


def encode_audio(autoencoder, path, sample_rate, device, model_type):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        if model_type in ("stft_vae", "stft_autoencoder"):
            latent, std = autoencoder.encode(audio_tensor)
        else:
            latent = autoencoder.encode(audio_tensor)
            std = None
    return {
        "latent": latent.squeeze(0).transpose(1, 0),
        "std": std.squeeze(0).transpose(1, 0) if std is not None else None,
        "num_samples": audio_tensor.shape[-1],
    }


def decode_latent(autoencoder, latent, std, model_type, num_samples, device):
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    latent = latent.to(device).transpose(2, 1)
    with torch.no_grad():
        if model_type in ("stft_vae", "stft_autoencoder"):
            if std is None:
                raise RuntimeError("STFT VAE decode requires std from the same evaluation pass")
            if std.ndim == 2:
                std = std.unsqueeze(0)
            audio = autoencoder.decode(latent, std.to(device).transpose(2, 1))
        else:
            audio = autoencoder.decode(latent)
    return audio[..., :num_samples]


def decode_std_for(cfg, source_item, mix_item, reference_item):
    decode_std = str(cfg.evaluation.extractor.get("decode_std", "mixture"))
    if decode_std == "source":
        return source_item["std"]
    if decode_std == "reference":
        return reference_item["std"]
    if decode_std == "mixture":
        return mix_item["std"]
    raise ValueError(f"Unsupported evaluation.extractor.decode_std: {decode_std}")


def make_compressor_runtime(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "device": device,
        "autoencoder": load_autoencoder(cfg).to(device),
        "model_type": cfg.compressor.model_type,
    }


def decode_compressor_output(cfg, pair, runtime):
    pair.pop("estimate", None)
    device = runtime["device"]
    autoencoder = runtime["autoencoder"]
    model_type = runtime["model_type"]
    encoded = encode_audio(autoencoder, pair["source"], cfg.evaluation.sample_rate, device, model_type)
    audio = decode_latent(
        autoencoder,
        encoded["latent"],
        encoded["std"],
        model_type,
        encoded["num_samples"],
        device,
    )
    pair["estimate_audio"] = tensor_to_audio_array(audio)
    return True


def load_autoencoder(cfg):
    ckpt = Path(cfg.save_dir) / "compressor" / cfg.compressor.checkpoint.ckpt_dir / "compressor.ckpt"
    if not ckpt.is_file():
        ckpt = Path(cfg.save_dir) / "compressor" / cfg.compressor.checkpoint.ckpt_dir / "last.ckpt"
    config = OmegaConf.to_container(cfg.compressor, resolve=True)
    model_type = config.get("model_type")
    if model_type in ("stft_vae", "stft_autoencoder"):
        from solospeech.vae_modules.stft_vae.models.stft_autoencoders import (
            create_autoencoder_from_config as create_stft_autoencoder_from_config,
        )

        model = create_stft_autoencoder_from_config(config)
    elif model_type == "stable_vae":
        from solospeech.vae_modules.stable_vae.models.autoencoders import (
            create_autoencoder_from_config as create_stable_autoencoder_from_config,
        )

        model = create_stable_autoencoder_from_config(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key[len("autoencoder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("autoencoder.")
    }
    if not state_dict:
        raise RuntimeError(f"No autoencoder.* weights found in checkpoint: {ckpt}")

    model.load_state_dict(state_dict)
    return model.eval()


def resolve_extractor_ckpt(cfg):
    if not is_null_value(cfg.evaluation.extractor.checkpoint_path):
        return Path(cfg.evaluation.extractor.checkpoint_path)
    candidates = [
        Path(cfg.save_dir) / "extractor" / "ckpts" / "extractor.pt",
        Path(cfg.save_dir) / "extractor" / "ckpts" / "last.ckpt",
        Path(cfg.save_dir) / "extractor" / "extractor.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_extractor(cfg):
    from solospeech.model.conditioners import SoloSpeech_TSE

    diffwrap = OmegaConf.to_container(cfg.extractor.diffwrap, resolve=True)
    model = SoloSpeech_TSE(
        copy.deepcopy(diffwrap["UDiT"]),
        copy.deepcopy(diffwrap["ViT"]),
    )
    checkpoint = torch.load(resolve_extractor_ckpt(cfg), map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return model.eval()


def make_extractor_scheduler(cfg):
    from diffusers import DDIMScheduler

    scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.extractor.ddim.diffusers, resolve=True))
    scheduler.set_timesteps(cfg.evaluation.extractor.num_infer_steps)
    return scheduler


def make_extractor_runtime(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "device": device,
        "autoencoder": load_autoencoder(cfg).to(device),
        "extractor": load_extractor(cfg).to(device),
        "scheduler": make_extractor_scheduler(cfg),
        "model_type": cfg.compressor.model_type,
    }


@torch.no_grad()
def sample_extractor_output(model, scheduler, mix_item, reference_item, cfg, device):
    mixture = mix_item["latent"].unsqueeze(0).to(device)
    reference = reference_item["latent"].unsqueeze(0).to(device)
    lengths = torch.LongTensor([mixture.shape[1]]).to(device)
    reference_lengths = torch.LongTensor([reference.shape[1]]).to(device)
    generator = torch.Generator(device=device).manual_seed(cfg.evaluation.extractor.seed)

    pred = torch.randn(mixture.shape, generator=generator, device=device)
    for timestep in scheduler.timesteps:
        model_input = scheduler.scale_model_input(pred, timestep)
        model_output, _ = model(
            x=model_input,
            timesteps=timestep,
            mixture=mixture,
            reference=reference,
            x_len=lengths,
            ref_len=reference_lengths,
        )
        pred = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=pred,
            eta=cfg.evaluation.extractor.eta,
            generator=generator,
        ).prev_sample
    return pred.squeeze(0).detach()


def generate_extractor_output(cfg, pair, runtime):
    pair.pop("estimate", None)
    if "mix" not in pair:
        raise RuntimeError(f"Extractor evaluation requires mix for {pair['id']}")

    device = runtime["device"]
    autoencoder = runtime["autoencoder"]
    extractor = runtime["extractor"]
    scheduler = runtime["scheduler"]
    model_type = runtime["model_type"]
    if "reference" not in pair:
        raise RuntimeError(f"Extractor evaluation requires enrollment/reference for {pair['id']}")
    reference_path = pair["reference"]

    source_item = encode_audio(autoencoder, pair["source"], cfg.evaluation.sample_rate, device, model_type)
    mix_item = encode_audio(autoencoder, pair["mix"], cfg.evaluation.sample_rate, device, model_type)
    reference_item = encode_audio(autoencoder, reference_path, cfg.evaluation.sample_rate, device, model_type)

    pred = sample_extractor_output(extractor, scheduler, mix_item, reference_item, cfg, device)

    audio = decode_latent(
        autoencoder,
        pred,
        decode_std_for(cfg, source_item, mix_item, reference_item),
        model_type,
        mix_item["num_samples"],
        device,
    )
    pair["estimate_audio"] = tensor_to_audio_array(audio)
    return True


def save_audio(path, sample_rate, audio):
    import torchaudio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().float().cpu().clamp(-0.999, 0.999)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


def save_audio_array(path, sample_rate, audio):
    audio = np.asarray(audio, dtype=np.float32)
    save_audio(path, sample_rate, torch.from_numpy(audio))


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def audio_sample_dir(cfg, category, pair):
    return get_evaluation_root(cfg) / "audio_samples" / category / cfg.evaluation.split / safe_name(pair["id"])


def maybe_limit_pairs(cfg, pairs, category):
    max_items = cfg.evaluation.max_items
    if max_items:
        pairs = pairs[:max_items]

    if not save_audio_outputs(cfg):
        return pairs

    total = len(pairs)
    num_samples = min(save_audio_num_samples(cfg), total)
    if num_samples <= 0 or total == 0:
        return []

    rng = random.Random(save_audio_seed(cfg))
    selected_indices = sorted(rng.sample(range(total), num_samples))
    selected_pairs = [pairs[idx] for idx in selected_indices]
    logger.info(f"Selected {len(selected_pairs)}/{total} random {category} samples for evaluation and audio saving")
    return selected_pairs


def save_demo_audio_bundle(cfg, category, pair):
    output_dir = audio_sample_dir(cfg, category, pair)
    sample_rate = cfg.evaluation.sample_rate
    save_audio_array(output_dir / "ground_truth.wav", sample_rate, load_audio(pair["source"], sample_rate))

    if pair.get("mix") is not None:
        save_audio_array(output_dir / "mixture.wav", sample_rate, load_audio(pair["mix"], sample_rate))
    if pair.get("reference") is not None:
        save_audio_array(output_dir / "enrollment.wav", sample_rate, load_audio(pair["reference"], sample_rate))
    if pair.get("estimate_audio") is not None:
        save_audio_array(output_dir / "extractor.wav", sample_rate, pair["estimate_audio"])
    if pair.get("system_audio") is not None:
        save_audio_array(output_dir / "final.wav", sample_rate, pair["system_audio"])


def resolve_corrector_ckpt(cfg):
    if not is_null_value(cfg.evaluation.corrector.checkpoint_path):
        return Path(cfg.evaluation.corrector.checkpoint_path)
    return Path(cfg.save_dir) / "corrector" / cfg.corrector.checkpoint.ckpt_dir / "corrector.ckpt"


def make_corrector_runtime(cfg):
    from solospeech.corrector.fastgeco.model import ScoreModel
    from solospeech.corrector.geco.util.other import pad_spec

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScoreModel.load_from_checkpoint(str(resolve_corrector_ckpt(cfg)), batch_size=1, num_workers=0, kwargs={"gpu": False})
    model.eval(no_ema=False)
    model.to(device)
    return {"device": device, "model": model, "pad_spec": pad_spec}


def run_corrector_output(cfg, pair, runtime):
    pair.pop("system_estimate", None)
    if "mix" not in pair or ("estimate_audio" not in pair and "estimate" not in pair):
        raise RuntimeError(
            f"Cannot run corrector for {pair['id']}: evaluation must have generated extractor estimate and manifest must provide mix."
        )

    device = runtime["device"]
    model = runtime["model"]
    pad_spec = runtime["pad_spec"]
    estimate_audio = pair.get("estimate_audio")
    if estimate_audio is None:
        estimate_audio = load_audio(pair["estimate"], cfg.evaluation.sample_rate)
    estimate = torch.from_numpy(estimate_audio).unsqueeze(0).to(device)
    mixture = torch.from_numpy(load_audio(pair["mix"], cfg.evaluation.sample_rate)).unsqueeze(0).to(device)
    length = min(estimate.shape[-1], mixture.shape[-1])
    estimate = estimate[..., :length]
    mixture = mixture[..., :length]
    norm_factor = mixture.abs().max().clamp_min(1e-8)
    estimate_norm = estimate / norm_factor
    mixture_norm = mixture / norm_factor

    y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(estimate_norm)), 0))
    m = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(mixture_norm)), 0))
    timesteps = torch.linspace(
        cfg.evaluation.corrector.reverse_starting_point,
        model.t_eps,
        cfg.evaluation.corrector.num_steps,
        device=device,
    )
    std = model.sde._std(cfg.evaluation.corrector.reverse_starting_point * torch.ones((m.shape[0],), device=device))
    x_t = m + torch.randn_like(m) * std[:, None, None, None]
    with torch.no_grad():
        for idx, timestep in enumerate(timesteps):
            dt = timestep - timesteps[idx + 1] if idx != len(timesteps) - 1 else timesteps[-1]
            f, g = model.sde.sde(x_t, timestep, m)
            vec_t = torch.ones(m.shape[0], device=device) * timestep
            score = model.forward(x_t, vec_t, m, y, vec_t[:, None, None, None])
            mean_x_tm1 = x_t - (f - g**2 * score) * dt
            if idx == len(timesteps) - 1:
                x_t = mean_x_tm1
            else:
                x_t = mean_x_tm1 + torch.randn_like(x_t) * g * torch.sqrt(dt)
        audio = model.to_audio(x_t.squeeze(), length)
        audio = audio * norm_factor / audio.abs().max().clamp_min(1e-8)
    pair["system_audio"] = tensor_to_audio_array(
        audio,
        label=f"corrector output for {pair['id']}",
    )
    return True


def compute_row(cfg, category, pair, metrics, dnsmos=None, wer=None, item_index=None, total_items=None):
    log_prefix = f"[{item_index}/{total_items}] " if item_index is not None and total_items is not None else ""
    sample_rate = cfg.evaluation.sample_rate
    if category in ("corrector", "system"):
        estimate_path = pair.get("system_estimate") or pair.get("estimate")
        generated_audio = pair.get("system_audio")
        if generated_audio is None:
            generated_audio = pair.get("estimate_audio")
    else:
        estimate_path = pair.get("estimate")
        generated_audio = pair.get("estimate_audio")
    if estimate_path is None and generated_audio is None:
        logger.warning(f"{log_prefix}Skip {pair['id']}: no estimate path for {category}")
        return None

    estimate_label = str(estimate_path) if estimate_path is not None else "memory"
    row = {"id": pair["id"], "source": str(pair["source"]), "estimate": estimate_label}
    reference_audio = None
    estimate_audio = None

    def ensure_audio():
        nonlocal reference_audio, estimate_audio
        if reference_audio is None:
            reference_audio = load_audio(pair["source"], sample_rate)
            estimate_audio = generated_audio if generated_audio is not None else load_audio(estimate_path, sample_rate)

    for metric in metrics:
        name = metric_name(metric)
        try:
            if name == "sisdr":
                ensure_audio()
                row[name] = si_sdr(estimate_audio, reference_audio)
            elif name == "pesq":
                ensure_audio()
                row[name] = compute_pesq(reference_audio, estimate_audio, sample_rate)
            elif name == "estoi":
                ensure_audio()
                row[name] = compute_estoi(reference_audio, estimate_audio, sample_rate)
            elif name == "dnsmos":
                if generated_audio is not None:
                    row.update(dnsmos.score_audio(generated_audio, sample_rate))
                else:
                    row.update(dnsmos(estimate_path, sample_rate))
            elif name == "wer":
                hypothesis_audio = generated_audio if generated_audio is not None else estimate_path
                row[name] = wer(pair["source"], hypothesis_audio, pair.get("transcript"))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        except Exception as exc:
            logger.warning(f"{log_prefix}{metric} failed for {estimate_label}: {exc}")
            row[name] = float("nan")

    logger.info(f"{log_prefix}{category} item {pair['id']}: {format_item_metrics(row)}")
    return row


def compute_rows(cfg, category, pairs, metrics, dnsmos=None, wer=None):
    rows = []
    total = len(pairs)
    for idx, pair in enumerate(pairs, start=1):
        row = compute_row(
            cfg,
            category,
            pair,
            metrics,
            dnsmos=dnsmos,
            wer=wer,
            item_index=idx,
            total_items=total,
        )
        if row is not None:
            rows.append(row)
    return rows


def run_category_pipeline(
    cfg,
    category,
    pairs,
    metrics,
    dnsmos=None,
    wer=None,
    compressor_runtime=None,
    extractor_runtime=None,
    corrector_runtime=None,
):
    rows = []
    total = len(pairs)
    if save_audio_outputs(cfg) and total > 0:
        sample_dir = get_evaluation_root(cfg) / "audio_samples" / category / cfg.evaluation.split
        logger.info(f"Saving {total} {category} audio samples to {sample_dir}")

    for idx, pair in enumerate(pairs, start=1):
        if category == "compressor":
            decode_compressor_output(cfg, pair, compressor_runtime)
        elif category == "extractor":
            generate_extractor_output(cfg, pair, extractor_runtime)
        elif category in ("corrector", "system"):
            generate_extractor_output(cfg, pair, extractor_runtime)
            run_corrector_output(cfg, pair, corrector_runtime)
        else:
            raise ValueError(f"Unsupported evaluation category: {category}")

        if save_audio_outputs(cfg):
            save_demo_audio_bundle(cfg, category, pair)

        row = compute_row(
            cfg,
            category,
            pair,
            metrics,
            dnsmos=dnsmos,
            wer=wer,
            item_index=idx,
            total_items=total,
        )
        if row is not None:
            rows.append(row)
    return rows


def write_results(cfg, category, rows):
    if save_audio_outputs(cfg):
        logger.info(f"Skip writing {category} result files because evaluation.save_audio_outputs=true")
        return

    output_dir = get_evaluation_root(cfg) / category
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning(f"No rows to write for {category}")
        return

    keys = sorted({key for row in rows for key in row})
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for key in keys:
        if key in ("id", "source", "estimate"):
            continue
        values = [row.get(key, float("nan")) for row in rows]
        summary[key] = {"mean": mean_std(values)[0], "std": mean_std(values)[1]}

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "summary.txt").open("w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value['mean']:.4f} +/- {value['std']:.4f}\n")

    logger.info(f"Wrote {category} results to {output_dir}")


def validate_decode_std_consistency(cfg, by_category):
    if "corrector" not in by_category and "system" not in by_category:
        return

    eval_std = str(cfg.evaluation.extractor.get("decode_std", "mixture"))
    train_std = str(cfg.corrector.generation.get("decode_std", "mixture"))
    if eval_std != train_std:
        raise ValueError(
            "evaluation.extractor.decode_std must match corrector.generation.decode_std "
            f"for corrector/system evaluation, got {eval_std!r} vs {train_std!r}."
        )


@hydra.main(
    version_base=None,
    config_path="../configs/tse",
    config_name="SoloSpeech",
)
def main(cfg: DictConfig):
    metrics = normalize_metrics(cfg.evaluation.metrics)
    logger.info(f"Evaluation metrics: {metrics}")
    by_category = defaultdict(list)
    for metric in metrics:
        by_category[category_for_metric(metric)].append(metric)
    validate_decode_std_consistency(cfg, by_category)

    dnsmos = DNSMOS(cfg.evaluation.dnsmos_num_threads) if any(metric.endswith("_dnsmos") for metric in metrics) else None
    wer = WERComputer(cfg) if any(metric.endswith("_wer") for metric in metrics) else None

    if "compressor" in by_category:
        compressor_pairs = maybe_limit_pairs(cfg, get_compressor_audio_pairs(cfg), "compressor")
        rows = run_category_pipeline(
            cfg,
            "compressor",
            compressor_pairs,
            by_category["compressor"],
            dnsmos=dnsmos,
            wer=wer,
            compressor_runtime=make_compressor_runtime(cfg),
        )
        write_results(cfg, "compressor", rows)

    if "extractor" in by_category:
        extractor_pairs = maybe_limit_pairs(cfg, get_extractor_audio_pairs(cfg), "extractor")
        rows = run_category_pipeline(
            cfg,
            "extractor",
            extractor_pairs,
            by_category["extractor"],
            dnsmos=dnsmos,
            wer=wer,
            extractor_runtime=make_extractor_runtime(cfg),
        )
        write_results(cfg, "extractor", rows)

    if "corrector" in by_category:
        corrector_pairs = maybe_limit_pairs(cfg, get_corrector_audio_pairs(cfg), "corrector")
        rows = run_category_pipeline(
            cfg,
            "corrector",
            corrector_pairs,
            by_category["corrector"],
            dnsmos=dnsmos,
            wer=wer,
            extractor_runtime=make_extractor_runtime(cfg),
            corrector_runtime=make_corrector_runtime(cfg),
        )
        write_results(cfg, "corrector", rows)

    if "system" in by_category:
        system_pairs = maybe_limit_pairs(cfg, get_system_audio_pairs(cfg), "system")
        rows = run_category_pipeline(
            cfg,
            "system",
            system_pairs,
            by_category["system"],
            dnsmos=dnsmos,
            wer=wer,
            extractor_runtime=make_extractor_runtime(cfg),
            corrector_runtime=make_corrector_runtime(cfg),
        )
        write_results(cfg, "system", rows)


if __name__ == "__main__":
    main()
