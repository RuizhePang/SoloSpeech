#!/usr/bin/env python3

import os
import random
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def absolute_data_config(cfg):
    data_cfg = OmegaConf.to_container(cfg.corrector.data, resolve=True)
    data_root = Path(cfg.data_dir)
    extractor_name = get_extractor_name()
    gen_cfg = cfg.corrector.generation
    default_root = (
        data_root
        / "Libri2Mix"
        / "extractor"
        / extractor_name
        / "Libri2Mix"
        / gen_cfg.sample_rate_dir
        / gen_cfg.mix_mode
    )
    default_splits = {
        "train_dir": "train",
        "val_dir": "dev",
        "test_dir": "test",
    }
    for key, split in default_splits.items():
        value = data_cfg.get(key)
        if value:
            path = Path(value)
            data_cfg[key] = str(path if path.is_absolute() else data_root / path)
        else:
            data_cfg[key] = str(default_root / split)
    return data_cfg


def get_extractor_name():
    from hydra.core.hydra_config import HydraConfig

    return HydraConfig.get().runtime.choices["extractor"]


def make_model(cfg):
    from solospeech.corrector.fastgeco.model import ScoreModel
    from solospeech.corrector.geco.data_module import SpecsDataModule

    corrector_cfg = cfg.corrector
    data_cfg = absolute_data_config(cfg)
    backbone_cfg = OmegaConf.to_container(corrector_cfg.backbone, resolve=True)
    sde_cfg = OmegaConf.to_container(corrector_cfg.sde, resolve=True)
    training_cfg = corrector_cfg.training
    model_cfg = corrector_cfg.model

    kwargs = {
        **data_cfg,
        **backbone_cfg,
        **sde_cfg,
        "gpus": training_cfg.num_gpus,
    }
    model = ScoreModel(
        backbone=model_cfg.backbone,
        sde=model_cfg.sde,
        lr=corrector_cfg.optimization.learning_rate,
        ema_decay=model_cfg.ema_decay,
        t_eps=model_cfg.t_eps,
        loss_abs_exponent=model_cfg.loss_abs_exponent,
        num_eval_files=model_cfg.num_eval_files,
        loss_type=training_cfg.loss_type,
        data_module_cls=SpecsDataModule,
        output_scale=model_cfg.output_scale,
        inference_N=training_cfg.inference_N,
        inference_start=training_cfg.inference_start,
        **kwargs,
    )
    model.add_para(
        training_cfg.N_min,
        training_cfg.N_max,
        training_cfg.t_rsp_min,
        training_cfg.t_rsp_max,
        data_cfg["batch_size"],
        training_cfg.loss_type,
        corrector_cfg.optimization.learning_rate,
        training_cfg.stop_iteration_random,
        training_cfg.inference_N,
        training_cfg.inference_start,
    )
    model.data_module.num_workers = data_cfg["num_workers"]
    model.data_module.gpu = torch.cuda.is_available()
    return model


def resolve_resume_path(ckpt_dir, resume_from):
    last_ckpt = Path(ckpt_dir) / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt
    if resume_from and Path(resume_from).is_file():
        return Path(resume_from)
    return None


def metric_value(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return value.detach().float().cpu().item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def make_loguru_metrics_callback(pl):
    class LoguruMetricsCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            value = metric_value(trainer.callback_metrics.get("train_loss_epoch"))
            if value is not None:
                logger.info(f"Epoch: [{trainer.current_epoch + 1}]    Train Loss: {value:.6f}")

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            parts = []
            for name in ("valid_loss", "pesq", "si_sdr", "estoi"):
                value = metric_value(metrics.get(name))
                if value is not None:
                    parts.append(f"{name}: {value:.6f}")
            if parts:
                logger.info(f"Epoch: [{trainer.current_epoch + 1}]    " + "    ".join(parts))

    return LoguruMetricsCallback()


@hydra.main(
    version_base=None,
    config_path="../../configs/tse",
    config_name="SoloSpeech",
)
def main(cfg: DictConfig):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.strategies import DDPStrategy

    corrector_cfg = cfg.corrector
    train_cfg = corrector_cfg.training
    save_dir = Path(cfg.save_dir)
    ckpt_dir = save_dir / corrector_cfg.checkpoint.ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)
    seed_everything(train_cfg.seed)

    model = make_model(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor=corrector_cfg.checkpoint.monitor,
        mode=corrector_cfg.checkpoint.mode,
        save_top_k=1,
        save_last=True,
        filename=corrector_cfg.checkpoint.filename,
        auto_insert_metric_name=False,
    )

    if train_cfg.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = train_cfg.strategy

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=train_cfg.num_gpus,
        num_nodes=train_cfg.num_nodes,
        strategy=strategy,
        precision=train_cfg.precision,
        max_epochs=train_cfg.max_epochs,
        max_steps=train_cfg.max_steps,
        log_every_n_steps=train_cfg.log_every_n_steps,
        num_sanity_val_steps=train_cfg.num_sanity_val_steps,
        gradient_clip_val=train_cfg.gradient_clip_val,
        callbacks=[checkpoint_callback, make_loguru_metrics_callback(pl)],
        logger=False,
        default_root_dir=str(save_dir),
        enable_progress_bar=False,
    )

    resume_path = resolve_resume_path(ckpt_dir, train_cfg.resume_from)
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")

    trainer.fit(model, ckpt_path=str(resume_path) if resume_path else None)


if __name__ == "__main__":
    main()
