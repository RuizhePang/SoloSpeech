#!/usr/bin/env python3

import copy
import os
import random
import shutil
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_compressor_name():
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.choices["compressor"]


def get_extractor_name():
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.choices["extractor"]


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
        raise RuntimeError(f"No autoencoder.* weights found in checkpoint: {ckpt_path}")

    model.load_state_dict(state_dict)
    return model.eval()


def load_extractor(cfg, ckpt_path):
    from solospeech.model.conditioners import SoloSpeech_TSE

    diffwrap = OmegaConf.to_container(cfg.extractor.diffwrap, resolve=True)
    model = SoloSpeech_TSE(
        copy.deepcopy(diffwrap["UDiT"]),
        copy.deepcopy(diffwrap["ViT"]),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return model.eval()


def resolve_compressor_ckpt_path(cfg):
    gen_cfg = cfg.corrector.generation
    if gen_cfg.compressor_ckpt_path:
        return Path(gen_cfg.compressor_ckpt_path).resolve()

    compressor_cfg = cfg.compressor
    if compressor_cfg.checkpoint.ckpt_path:
        return Path(compressor_cfg.checkpoint.ckpt_path).resolve()

    experiment_dir = Path(cfg.save_dir)
    candidates = [
        experiment_dir / "compressor" / compressor_cfg.checkpoint.ckpt_dir / "compressor.ckpt",
        experiment_dir / "compressor" / compressor_cfg.checkpoint.ckpt_dir / "last.ckpt",
        experiment_dir / "compressor" / "compressor.ckpt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_extractor_ckpt_path(cfg):
    gen_cfg = cfg.corrector.generation
    if gen_cfg.extractor_ckpt_path:
        return Path(gen_cfg.extractor_ckpt_path).resolve()

    experiment_dir = Path(cfg.save_dir)
    candidates = [
        experiment_dir / "extractor" / "ckpts" / "extractor.pt",
        experiment_dir / "extractor" / "ckpts" / "last.ckpt",
        experiment_dir / "extractor" / "extractor.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def make_dataset(cfg, compressor_name, subset):
    from solospeech.dataset.tse import TSEDataset

    data_cfg = cfg.extractor.data
    gen_cfg = cfg.corrector.generation
    base_dir = Path(cfg.data_dir) / "Libri2Mix" / "LibriMixData"
    vae_dir = Path(cfg.data_dir) / "Libri2Mix" / "compressor" / compressor_name
    csv_dir = (
        Path(cfg.data_dir)
        / "Libri2Mix"
        / "SpeakerBeamData"
        / gen_cfg.sample_rate_dir
        / gen_cfg.mix_mode
        / subset
    )

    return TSEDataset(
        csv_dir=str(csv_dir),
        base_dir=str(base_dir),
        vae_dir=str(vae_dir),
        task=data_cfg.task,
        sample_rate=data_cfg.sample_rate,
        vae_rate=data_cfg.vae_rate,
        n_src=data_cfg.n_src,
        min_length=data_cfg.min_length,
        debug=gen_cfg.debug,
        training=False,
    )


def make_loader(cfg, compressor_name, subsets):
    datasets = [make_dataset(cfg, compressor_name, subset) for subset in subsets]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset,
        num_workers=cfg.corrector.generation.num_workers,
        batch_size=cfg.corrector.generation.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=datasets[0].collate,
    )


def source_index(source_path):
    path = str(source_path)
    if "/s1/" in path:
        return 1
    if "/s2/" in path:
        return 2
    raise RuntimeError(f"Cannot infer source index from path: {source_path}")


def output_paths(output_dir, mixture_path, source_path):
    stem = Path(mixture_path).stem
    idx = source_index(source_path)
    return {
        "mix": Path(output_dir) / f"{stem}_mix.wav",
        "source": Path(output_dir) / f"{stem}_source{idx}.wav",
        "hatp": Path(output_dir) / f"{stem}_source{idx}hatP.wav",
    }


def copy_audio(src_path, dst_path, skip_existing):
    dst_path = Path(dst_path)
    if skip_existing and dst_path.is_file():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


def save_audio(path, sample_rate, audio):
    import torchaudio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().float().cpu().clamp(-0.999, 0.999)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


@torch.no_grad()
def sample_extractor(model, scheduler, batch, cfg, device):
    gen_cfg = cfg.corrector.generation
    mixture = batch["mixture_vae"].to(device)
    reference = batch["reference_vae"].to(device)
    lengths = batch["length"].to(device)
    reference_lengths = batch["reference_length"].to(device)

    generator = torch.Generator(device=device).manual_seed(gen_cfg.seed)
    scheduler.set_timesteps(gen_cfg.num_infer_steps)
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
            eta=gen_cfg.eta,
            generator=generator,
        ).prev_sample
    return pred


def decode_prediction(autoencoder, latent, batch, cfg, device):
    compressor_cfg = cfg.compressor
    model_type = compressor_cfg.model_type
    gen_cfg = cfg.corrector.generation

    if model_type in ("stft_vae", "stft_autoencoder"):
        std_key = f"{gen_cfg.decode_std}_std"
        std = batch[std_key]
        if std is None:
            raise RuntimeError("Stage5 requires std saved by stage3. Rerun stage3 to regenerate VAE .pt files.")
        return autoencoder.decode(latent.transpose(2, 1), std.to(device).transpose(2, 1))
    return autoencoder.decode(latent.transpose(2, 1))


def generate_split(model, autoencoder, scheduler, loader, split_name, output_dir, cfg, device):
    gen_cfg = cfg.corrector.generation
    sample_rate = cfg.extractor.data.sample_rate
    hop = cfg.extractor.data.sample_rate // cfg.extractor.data.vae_rate
    total_items = 0
    saved_items = 0

    for batch in tqdm(loader, desc=f"Generate {split_name}"):
        item_paths = [
            output_paths(output_dir, mixture_path, source_path)
            for mixture_path, source_path in zip(batch["mixture_path"], batch["source_path"])
        ]
        for paths, mixture_path, source_path in zip(item_paths, batch["mixture_path"], batch["source_path"]):
            copy_audio(mixture_path, paths["mix"], gen_cfg.skip_existing)
            copy_audio(source_path, paths["source"], gen_cfg.skip_existing)

        total_items += len(item_paths)
        if gen_cfg.skip_existing and all(paths["hatp"].is_file() for paths in item_paths):
            continue

        latent = sample_extractor(model, scheduler, batch, cfg, device)
        pred_wav = decode_prediction(autoencoder, latent, batch, cfg, device)

        for item_idx, paths in enumerate(item_paths):
            wav_length = int(batch["length"][item_idx].item()) * hop
            save_audio(paths["hatp"], sample_rate, pred_wav[item_idx, :, :wav_length])
            saved_items += 1

    logger.info(f"{split_name}: processed {total_items} items, generated {saved_items} hatP files")


@hydra.main(
    version_base=None,
    config_path="../configs/tse",
    config_name="SoloSpeech",
)
def main(cfg: DictConfig):
    from diffusers import DDIMScheduler

    seed_everything(cfg.corrector.generation.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compressor_name = get_compressor_name()
    extractor_name = get_extractor_name()
    compressor_cfg = cfg.compressor
    compressor_config = OmegaConf.to_container(compressor_cfg, resolve=True)
    compressor_ckpt = resolve_compressor_ckpt_path(cfg)
    extractor_ckpt = resolve_extractor_ckpt_path(cfg)

    logger.info(f"Loading compressor checkpoint: {compressor_ckpt}")
    autoencoder = load_autoencoder(compressor_config, compressor_ckpt).to(device)
    logger.info(f"Loading extractor checkpoint: {extractor_ckpt}")
    model = load_extractor(cfg, extractor_ckpt).to(device)
    scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.extractor.ddim.diffusers, resolve=True))

    output_root = (
        Path(cfg.data_dir)
        / "Libri2Mix"
        / "extractor"
        / extractor_name
        / "Libri2Mix"
        / cfg.corrector.generation.sample_rate_dir
        / cfg.corrector.generation.mix_mode
    )
    logger.info(f"Output root: {output_root}")
    for split_name, subsets in cfg.corrector.generation.subsets.items():
        output_dir = output_root / split_name
        os.makedirs(output_dir, exist_ok=True)
        loader = make_loader(cfg, compressor_name, list(subsets))
        generate_split(model, autoencoder, scheduler, loader, split_name, output_dir, cfg, device)


if __name__ == "__main__":
    main()
