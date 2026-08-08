import os
import random

import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import (
    load_ckpt_state_dict,
    remove_weight_norm_from_model,
)
from stable_audio_tools.training import (
    create_training_wrapper_from_config,
    create_demo_callback_from_config,
)
from stable_audio_tools.training.utils import copy_state_dict


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


@hydra.main(
    version_base=None,
    config_path="../../configs/tse",
    config_name="default",
)
def main(cfg: DictConfig):

    save_dir = cfg.save_dir
    data_dir = cfg.data_dir

    cfg.vae.checkpoint.ckpt_dir = os.path.join(save_dir, cfg.vae.checkpoint.ckpt_dir)
    for dataset in cfg.vae.dataset.datasets:
        dataset.path = os.path.join(data_dir, dataset.path)

    vae_cfg = cfg.vae

    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------
    seed = vae_cfg.training.seed

    # Different seed for each SLURM process
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ["SLURM_PROCID"])

    random.seed(seed)
    torch.manual_seed(seed)

    torch.set_float32_matmul_precision("high")

    # ---------------------------------------------------------
    # Stable Audio Tools config
    # ---------------------------------------------------------
    #
    # Convert Hydra DictConfig -> Python dict because
    # stable_audio_tools expects a normal config dictionary.
    #
    model_config = OmegaConf.to_container(vae_cfg, resolve=True)

    # These are our Hydra/runtime-only configs.
    # stable_audio_tools does not need them.
    dataset_config = model_config.pop("dataset", None)
    checkpoint_config = model_config.pop("checkpoint", None)
    model_config.pop("name", None)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=vae_cfg.training.batch_size,
        num_workers=vae_cfg.training.num_workers,
        sample_rate=vae_cfg.sample_rate,
        sample_size=vae_cfg.sample_size,
        audio_channels=vae_cfg.get("audio_channels", 2),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = create_model_from_config(model_config)

    # ---------------------------------------------------------
    # Load checkpoints
    # ---------------------------------------------------------
    if vae_cfg.checkpoint.pretrained_ckpt_path:
        pretrained_ckpt_path = to_absolute_path(vae_cfg.checkpoint.pretrained_ckpt_path)
        copy_state_dict(model, load_ckpt_state_dict(pretrained_ckpt_path))

    if (vae_cfg.checkpoint.remove_pretransform_weight_norm == "pre_load"):
        remove_weight_norm_from_model(model.pretransform)

    if vae_cfg.checkpoint.pretransform_ckpt_path:
        pretransform_ckpt_path = to_absolute_path(vae_cfg.checkpoint.pretransform_ckpt_path)

        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path))

    if (vae_cfg.checkpoint.remove_pretransform_weight_norm == "post_load"):
        remove_weight_norm_from_model(model.pretransform)

    # ---------------------------------------------------------
    # Training wrapper
    # ---------------------------------------------------------
    training_wrapper = create_training_wrapper_from_config(model_config, model)

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    exc_callback = ExceptionCallback()

    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    # ---------------------------------------------------------
    # Checkpoint directory
    # ---------------------------------------------------------
    if vae_cfg.checkpoint.ckpt_dir:
        ckpt_dir = to_absolute_path(vae_cfg.checkpoint.ckpt_dir)
    else:
        ckpt_dir = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        monitor="epoch/mrstft_loss",
        mode="min",
        save_top_k=1,
        filename="compressor",
        auto_insert_metric_name=False,
        save_last=True,
    )

    # ---------------------------------------------------------
    # Multi-GPU strategy
    # ---------------------------------------------------------
    if vae_cfg.training.strategy:

        if vae_cfg.training.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy

            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True,
            )

        else:
            strategy = vae_cfg.training.strategy

    else:
        strategy = (
            "ddp_find_unused_parameters_true"
            if vae_cfg.training.num_gpus > 1
            else "auto"
        )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------
    tb_logger = TensorBoardLogger(save_dir, name="tensorboard")
    trainer = pl.Trainer(
        devices=vae_cfg.training.num_gpus,
        accelerator="gpu",
        num_nodes=vae_cfg.training.num_nodes,
        strategy=strategy,
        precision=vae_cfg.training.precision,
        accumulate_grad_batches=(vae_cfg.training.accum_batches),
        callbacks=[
            ckpt_callback,
            demo_callback,
            exc_callback,
            save_model_config_callback,
        ],
        logger=tb_logger,
        log_every_n_steps=10,
        max_steps=vae_cfg.training.max_steps,
        default_root_dir=save_dir,
        gradient_clip_val=(vae_cfg.training.gradient_clip_val),
        reload_dataloaders_every_n_epochs=0,
        enable_progress_bar=False,
    )

    # ---------------------------------------------------------
    # Resume checkpoint
    # ---------------------------------------------------------
    ckpt_path = None

    if vae_cfg.checkpoint.ckpt_path:
        ckpt_path = to_absolute_path(vae_cfg.checkpoint.ckpt_path)

    elif ckpt_dir is not None:
        last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.isfile(last_ckpt_path):
            ckpt_path = last_ckpt_path
            logger.info(f"Resuming from checkpoint: {ckpt_path}")

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    trainer.fit(
        training_wrapper,
        train_dl,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
