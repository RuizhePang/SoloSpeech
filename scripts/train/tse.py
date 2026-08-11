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


def masked_mse_loss(predictions, targets, mask=None):
    if mask is None:
        return ((predictions - targets) ** 2).mean()

    mask = mask.unsqueeze(-1).long()
    mse = (predictions - targets) ** 2
    return (mse * mask).sum() / mask.sum().clamp_min(1)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_dataset(cfg, vae_name, subset, training):
    from solospeech.dataset.tse import TSEDataset

    data_cfg = cfg.extractor.data
    base_dir = Path(cfg.data_dir) / "Libri2Mix" / "LibriMixData"
    vae_dir = Path(cfg.data_dir) / "Libri2Mix" / vae_name
    csv_dir = (
        Path(cfg.data_dir)
        / "Libri2Mix"
        / "SpeakerBeamData"
        / data_cfg.sample_rate_dir
        / data_cfg.mix_mode
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
        debug=data_cfg.debug_train if training else data_cfg.debug_val,
        training=training,
    )


def make_train_loader(cfg, vae_name):
    train_sets = [
        make_dataset(cfg, vae_name=vae_name, subset=subset, training=True)
        for subset in cfg.extractor.data.train_subsets
    ]
    train_set = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    return DataLoader(
        train_set,
        num_workers=cfg.extractor.training.num_workers,
        batch_size=cfg.extractor.training.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_sets[0].collate,
    )


def make_val_loader(cfg, vae_name):
    val_set = make_dataset(
        cfg,
        vae_name=vae_name,
        subset=cfg.extractor.data.val_subset,
        training=False,
    )
    return DataLoader(
        val_set,
        num_workers=cfg.extractor.training.num_workers,
        batch_size=cfg.extractor.training.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=val_set.collate,
    )


def make_model(cfg):
    from solospeech.model.conditioners import SoloSpeech_TSE

    diffwrap = OmegaConf.to_container(cfg.extractor.diffwrap, resolve=True)
    return SoloSpeech_TSE(
        copy.deepcopy(diffwrap["UDiT"]),
        copy.deepcopy(diffwrap["ViT"]),
    )


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


def resolve_vae_ckpt_path(cfg):
    if cfg.vae.checkpoint.ckpt_path:
        return Path(cfg.vae.checkpoint.ckpt_path).resolve()

    extractor_dir = Path(cfg.save_dir)
    experiment_dir = extractor_dir.parent
    candidates = [
        experiment_dir / "compressor" / cfg.vae.checkpoint.ckpt_dir / "compressor.ckpt",
        experiment_dir / "compressor" / "compressor.ckpt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def make_optimizer(model, cfg):
    opt_cfg = cfg.extractor.optimization
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.learning_rate,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        weight_decay=opt_cfg.weight_decay,
        eps=opt_cfg.adam_epsilon,
    )


def train_step(model, batch, noise_scheduler, accelerator, v_prediction):
    mixture = batch["mixture_vae"].to(accelerator.device)
    target = batch["source_vae"].to(accelerator.device)
    reference = batch["reference_vae"].to(accelerator.device)
    lengths = batch["length"].to(accelerator.device)
    reference_lengths = batch["reference_length"].to(accelerator.device)

    noise = torch.randn(target.shape, device=accelerator.device)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (noise.shape[0],),
        device=accelerator.device,
    ).long()
    noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
    pred, pred_mask = model(
        x=noisy_target,
        timesteps=timesteps,
        mixture=mixture,
        reference=reference,
        x_len=lengths,
        ref_len=reference_lengths,
    )

    if v_prediction:
        target_value = noise_scheduler.get_velocity(target, noise, timesteps)
    else:
        target_value = noise
    return masked_mse_loss(pred, target_value, pred_mask)


@torch.no_grad()
def validate(model, val_loader, noise_scheduler, accelerator, cfg):
    model.eval()
    losses = []
    v_prediction = cfg.extractor.ddim.v_prediction
    for batch in tqdm(val_loader, disable=not accelerator.is_main_process, desc="Valid"):
        loss = train_step(model, batch, noise_scheduler, accelerator, v_prediction)
        losses.append(accelerator.gather_for_metrics(loss.detach()).mean())
    model.train()

    if not losses:
        return None
    return torch.stack(losses).mean().item()


def save_audio(path, sample_rate, audio):
    import torchaudio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().float().cpu().clamp(-0.999, 0.999)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(path), audio, sample_rate)


def copy_audio(src_path, dst_path):
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


@torch.no_grad()
def run_audio_demo(model, autoencoder, scheduler, demo_loader, cfg, accelerator, epoch):
    demo_cfg = cfg.extractor.demo
    if not demo_cfg.enabled:
        return

    model.eval()
    scheduler.set_timesteps(cfg.extractor.training.num_infer_steps)
    generator = torch.Generator(device=accelerator.device).manual_seed(demo_cfg.seed)
    demo_root = Path(cfg.save_dir) / "demo" / f"epoch_{epoch}"
    sample_rate = cfg.extractor.data.sample_rate
    hop = cfg.extractor.data.sample_rate // cfg.extractor.data.vae_rate
    vae_model_type = cfg.vae.model_type

    for batch_idx, batch in enumerate(tqdm(demo_loader, disable=not accelerator.is_main_process, desc="Demo")):
        if batch_idx >= demo_cfg.max_batches:
            break

        mixture = batch["mixture_vae"].to(accelerator.device)
        reference = batch["reference_vae"].to(accelerator.device)
        lengths = batch["length"].to(accelerator.device)
        reference_lengths = batch["reference_length"].to(accelerator.device)

        pred = torch.randn(mixture.shape, generator=generator, device=accelerator.device)
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
                eta=demo_cfg.eta,
                generator=generator,
            ).prev_sample

        source_std = batch["source_std"]
        if source_std is None and vae_model_type in ("stft_vae", "stft_autoencoder"):
            raise RuntimeError("Audio demo requires std saved by stage3. Rerun stage3 to regenerate VAE .pt files.")

        if vae_model_type in ("stft_vae", "stft_autoencoder"):
            pred_wav = autoencoder.decode(pred.transpose(2, 1), source_std.to(accelerator.device).transpose(2, 1))
        else:
            pred_wav = autoencoder.decode(pred.transpose(2, 1))

        if not accelerator.is_main_process:
            continue

        batch_size = pred_wav.shape[0]
        for item_idx in range(batch_size):
            mix_id = batch["id"][item_idx]
            item_root = demo_root / f"batch_{batch_idx:04d}" / mix_id
            wav_length = int(lengths[item_idx].item()) * hop
            save_audio(item_root / "pred.wav", sample_rate, pred_wav[item_idx, :, :wav_length])
            copy_audio(batch["mixture_path"][item_idx], item_root / "mixture.wav")
            copy_audio(batch["source_path"][item_idx], item_root / "source.wav")
            copy_audio(batch["reference_path"][item_idx], item_root / "reference.wav")
            copy_audio(batch["exclude_path"][item_idx], item_root / "exclude.wav")

    model.train()


@hydra.main(
    version_base=None,
    config_path="../../configs/tse",
    config_name="SoloSpeech",
)
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig
    from accelerate import Accelerator
    from diffusers import DDIMScheduler

    vae_name = HydraConfig.get().runtime.choices["vae"]
    train_cfg = cfg.extractor.training
    save_dir = Path(cfg.save_dir)
    ckpt_dir = save_dir / "ckpts"

    os.makedirs(save_dir / "demo", exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(train_cfg.seed)
    torch.set_num_threads(train_cfg.num_threads)

    accelerator = Accelerator(mixed_precision=train_cfg.amp)
    model = make_model(cfg)
    optimizer = make_optimizer(model, cfg)
    train_loader = make_train_loader(cfg, vae_name)
    val_loader = make_val_loader(cfg, vae_name)
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.extractor.ddim.diffusers, resolve=True))
    demo_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.extractor.ddim.diffusers, resolve=True))
    vae_config = OmegaConf.to_container(cfg.vae, resolve=True)
    autoencoder = load_autoencoder(vae_config, resolve_vae_ckpt_path(cfg))

    total = sum(param.nelement() for param in model.parameters())
    if accelerator.is_main_process:
        logger.info(f"Number of parameter: {total / 1e6:.2f}M")

    resume_from = train_cfg.resume_from
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"] + 1
        if accelerator.is_main_process:
            logger.info(f"Resuming from checkpoint: {resume_from}, starting from epoch {start_epoch}.")
    else:
        global_step = 0
        start_epoch = 0

    model, optimizer, train_loader, val_loader, autoencoder = accelerator.prepare(
        model, optimizer, train_loader, val_loader, autoencoder
    )

    running_loss = 0.0
    v_prediction = cfg.extractor.ddim.v_prediction

    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}")):
            loss = train_step(model, batch, noise_scheduler, accelerator, v_prediction)

            if torch.isnan(loss).item():
                if accelerator.is_main_process:
                    logger.warning(
                        f"Epoch: [{epoch + 1}][{train_cfg.epochs}]    "
                        f"Batch: [{step + 1}][{len(train_loader)}]  Nan  Loss"
                    )
                torch.cuda.empty_cache()
                continue

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += loss.item()

            if accelerator.is_main_process and global_step % train_cfg.log_step == 0:
                logger.info(
                    "Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}".format(
                        epoch + 1,
                        train_cfg.epochs,
                        step + 1,
                        len(train_loader),
                        running_loss / train_cfg.log_step,
                    ),
                )
                running_loss = 0.0

        if (epoch + 1) % train_cfg.validate_every == 0:
            val_loss = validate(model, val_loader, noise_scheduler, accelerator, cfg)
            if accelerator.is_main_process and val_loss is not None:
                logger.info(f"Epoch: [{epoch + 1}][{train_cfg.epochs}]    Val Loss: {val_loss:.6f}")

        if (epoch + 1) % cfg.extractor.demo.every == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_autoencoder = accelerator.unwrap_model(autoencoder)
            run_audio_demo(
                unwrapped_model,
                unwrapped_autoencoder,
                demo_scheduler,
                val_loader,
                cfg,
                accelerator,
                epoch + 1,
            )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and (epoch + 1) % train_cfg.save_every == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(
                {
                    "model": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                ckpt_dir / f"{epoch}.pt",
            )
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
