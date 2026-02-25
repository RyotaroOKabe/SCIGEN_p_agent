import os
from pathlib import Path
from typing import Optional

import dotenv
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """Safely read an environment variable.
    
    Reads an environment variable and returns its value. Raises errors if
    the variable is not defined or is empty (unless a default is provided).
    
    Args:
        env_name: The name of the environment variable to read.
        default: Optional default value to return if variable is not set or empty.
    
    Returns:
        The value of the environment variable, or the default if provided.
    
    Raises:
        KeyError: If environment variable is not defined and no default is provided.
        ValueError: If environment variable is empty and no default is provided.
    
    Example:
        >>> project_root = get_env("PROJECT_ROOT", "/default/path")
        >>> required_var = get_env("REQUIRED_VAR")  # Raises if not set
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(
                f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """Load environment variables from a .env file.
    
    Loads environment variables from a .env file using python-dotenv.
    This is equivalent to `source env_file` in bash. If no file is specified,
    searches for a .env file in the project root, current directory, or
    parent directories.
    
    Args:
        env_file: Path to .env file. If None, searches for .env file in:
            1. Project root (where this file is located)
            2. Current working directory
            3. Parent directories (up to 3 levels)
    
    Note:
        Environment variables are loaded with override=True, meaning existing
        environment variables will be overwritten by values in the .env file.
    
    Example:
        Load from default location:
        >>> load_envs()
        
        Load from specific file:
        >>> load_envs("/path/to/.env")
    """
    if env_file is None:
        # Try to find .env file in the project root
        # First, try the directory where this file is located (project root)
        script_dir = Path(__file__).parent.parent.parent
        env_file = script_dir / ".env"
        
        # If not found, try current directory
        if not env_file.exists():
            current_dir = Path.cwd()
            env_file = current_dir / ".env"
        
        # If still not found, try parent directories up to 3 levels
        if not env_file.exists():
            for parent in Path.cwd().parents[:3]:
                potential_env = parent / ".env"
                if potential_env.exists():
                    env_file = potential_env
                    break
        
        env_file = str(env_file) if isinstance(env_file, Path) else env_file
    
    if env_file and Path(env_file).exists():
        dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"


# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """Log hyperparameters to all configured loggers.
    
    Converts Hydra configuration to a dictionary and logs it along with
    model parameter statistics to all configured loggers (e.g., WandB).
    After logging, disables further hyperparameter logging to prevent
    duplicate entries.
    
    Args:
        cfg: Complete Hydra configuration object.
        model: PyTorch Lightning model instance.
        trainer: PyTorch Lightning trainer instance.
    
    What is logged:
        - All configuration parameters from cfg
        - Total number of model parameters
        - Number of trainable parameters
        - Number of non-trainable parameters
    
    Note:
        After calling this function, trainer.logger.log_hyperparams is
        disabled to prevent duplicate logging.
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel()
                                               for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)
