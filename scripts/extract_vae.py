#!/usr/bin/env python3

import math
from loguru import logger
from pathlib import Path

import hydra
import librosa
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

def load_autoencoder(config, ckpt_path):
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

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key[len("autoencoder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("autoencoder.")
    }
    if not state_dict:
        raise RuntimeError(
            f"No autoencoder.* weights found in checkpoint: {ckpt_path}"
        )

    model.load_state_dict(state_dict)
    return model.eval()


def iter_wavs(input_root):
    wavs = sorted(input_root.rglob("*.wav"))
    return wavs


def output_path_for(wav_path, relative_root, output_root):
    rel_path = wav_path.relative_to(relative_root)
    return output_root / rel_path.with_suffix(".pt")


def encode_wav(autoencoder, wav_path, sample_rate, device, model_type):
    audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
    audio = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type in ("stft_vae", "stft_autoencoder"):
            latent, _std = autoencoder.encode(audio)
        else:
            latent = autoencoder.encode(audio)

    return latent.squeeze(0).cpu()


def update_stats(latent, stats):
    values = latent.float()
    stats["sum"] += values.sum().item()
    stats["sum_sq"] += values.square().sum().item()
    stats["count"] += values.numel()


def print_stats(stats):
    if stats["count"] == 0:
        print("No latent values were encoded; shift/scale unavailable.")
        return

    mean = stats["sum"] / stats["count"]
    variance = max(stats["sum_sq"] / stats["count"] - mean * mean, 0.0)
    std = math.sqrt(variance)
    scale = 1.0 / std if std > 0 else float("inf")
    logger.info(f"shift: {-mean}")
    logger.info(f"scale: {scale}")


@hydra.main(
    version_base=None,
    config_path="../configs/tse",
    config_name="SoloSpeech",
)
def main(cfg: DictConfig):
    vae_name = HydraConfig.get().runtime.choices["vae"]

    relative_root = (Path(cfg.data_dir) / "Libri2Mix" / "LibriMixData").resolve()
    input_root = (relative_root / "Libri2Mix" / "wav16k").resolve()
    output_root = (Path(cfg.data_dir) / "Libri2Mix" / vae_name).resolve()
    vae_ckpt_path = (
        Path(cfg.save_dir) / cfg.vae.checkpoint.ckpt_dir / "compressor.ckpt"
    ).resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing input root: {input_root}")
    if not vae_ckpt_path.is_file():
        raise FileNotFoundError(f"Missing VAE checkpoint: {vae_ckpt_path}")

    vae_config = OmegaConf.to_container(cfg.vae, resolve=True)
    sample_rate = int(vae_config["sample_rate"])
    model_type = vae_config["model_type"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Input root: {input_root}")
    logger.info(f"Relative root: {relative_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"VAE config: {vae_name}")
    logger.info(f"VAE checkpoint: {vae_ckpt_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Sample rate: {sample_rate}")

    autoencoder = load_autoencoder(vae_config, vae_ckpt_path).to(device)
    wavs = iter_wavs(input_root)
    if not wavs:
        raise RuntimeError(f"No wav files found under {input_root}")

    stats = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    encoded = 0
    skipped = 0

    for wav_path in wavs:
        logger.info(f"Encoding idx {encoded + skipped + 1}/{len(wavs)}: {wav_path}")
        out_path = output_path_for(wav_path, relative_root, output_root)
        if out_path.exists():
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        latent = encode_wav(autoencoder, wav_path, sample_rate, device, model_type)
        torch.save(latent, out_path)
        update_stats(latent, stats)
        encoded += 1

    logger.info(f"Encoded files: {encoded}")
    logger.info(f"Skipped existing files: {skipped}")
    print_stats(stats)


if __name__ == "__main__":
    main()
