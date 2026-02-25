from pathlib import Path
from typing import List
import sys
sys.path.append('.')
import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from scigen.common.utils import log_hyperparameters, PROJECT_ROOT

import wandb



def build_callbacks(cfg: DictConfig) -> List[Callback]:
    """Build list of PyTorch Lightning callbacks based on configuration.
    
    Creates callbacks for learning rate monitoring, early stopping, and model
    checkpointing based on the provided configuration.
    
    Args:
        cfg: Hydra configuration object containing logging and training settings.
            Expected keys:
            - cfg.logging.lr_monitor: Learning rate monitor settings
            - cfg.train.early_stopping: Early stopping settings
            - cfg.train.model_checkpoints: Model checkpoint settings
            - cfg.train.monitor_metric: Metric to monitor
            - cfg.train.monitor_metric_mode: 'min' or 'max'
    
    Returns:
        List of configured PyTorch Lightning Callback objects.
    """
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )

    return callbacks


def run(cfg: DictConfig) -> None:
    """Main training function that orchestrates the complete training process.
    
    This function handles:
    1. Setting up random seeds for reproducibility
    2. Configuring debug mode if enabled
    3. Instantiating data module and model
    4. Setting up callbacks (early stopping, checkpointing, LR monitoring)
    5. Configuring WandB logger
    6. Creating PyTorch Lightning Trainer
    7. Running training, validation, and testing
    
    Args:
        cfg: Complete Hydra configuration object containing all training parameters.
            Must include:
            - cfg.train: Training settings (deterministic, random_seed, pl_trainer, etc.)
            - cfg.data: Data configuration (datamodule settings)
            - cfg.model: Model configuration (architecture, hyperparameters)
            - cfg.optim: Optimizer configuration
            - cfg.logging: Logging configuration (WandB, callbacks)
    
    Raises:
        FileNotFoundError: If required data files are not found.
        RuntimeError: If CUDA is not available when GPU training is requested.
    
    Example:
        Configuration is typically provided by Hydra decorator:
        
        @hydra.main(config_path="conf", config_name="default")
        def main(cfg):
            run(cfg)
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.pl_trainer.accelerator = "cpu"
        cfg.train.pl_trainer.devices = 1
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        train=cfg.train,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')
    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="thread"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Load checkpoint (if exist)
    ckpts = list(hydra_dir.glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
    else:
        ckpt = None
          
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        **cfg.train.pl_trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Main entry point for training script.
    
    This function is decorated with Hydra to automatically load and parse
    configuration files from the conf/ directory. The configuration is then
    passed to the run() function to start training.
    
    Args:
        cfg: Hydra configuration object automatically loaded from conf/default.yaml
            and any command-line overrides.
    
    Example:
        Run from command line:
        
        # Basic training
        python scigen/run.py data=mp_20 model=diffusion_w_type
        
        # With overrides
        python scigen/run.py data=alex_2d model=diffusion_w_type \\
            train.pl_trainer.max_epochs=500
    """
    run(cfg)


if __name__ == "__main__":
    main()
