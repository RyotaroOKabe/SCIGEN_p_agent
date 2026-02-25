import random
from typing import Optional, Sequence
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

from scigen.common.utils import PROJECT_ROOT
from scigen.common.data_utils import get_scaler_from_data_list


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for crystal structure datasets.
    
    This module handles loading, preprocessing, and batching of crystal structure
    data for training, validation, and testing. It automatically computes or loads
    data scalers for normalization.
    
    Attributes:
        datasets: Configuration for train/val/test datasets
        num_workers: Number of data loading workers per split
        batch_size: Batch sizes for each split
        train_dataset: Training dataset (set in setup())
        val_datasets: Validation datasets (set in setup())
        test_datasets: Test datasets (set in setup())
        lattice_scaler: Scaler for lattice parameters
        scaler: Scaler for target property (e.g., formation energy)
    """
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        """Initialize the data module.
        
        Args:
            datasets: Hydra configuration for datasets. Must contain:
                - datasets.train: Training dataset configuration
                - datasets.val: List of validation dataset configurations
                - datasets.test: List of test dataset configurations
            num_workers: Configuration dict with keys 'train', 'val', 'test'
                specifying number of data loading workers
            batch_size: Configuration dict with keys 'train', 'val', 'test'
                specifying batch sizes
            scaler_path: Optional path to pre-computed scalers. If None,
                scalers are computed from training data.
        """
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path: Optional[str]) -> None:
        """Compute or load data scalers for normalization.
        
        If scaler_path is provided, loads pre-computed scalers from disk.
        Otherwise, computes scalers from the training dataset.
        
        Args:
            scaler_path: Path to directory containing saved scalers, or None
                to compute from training data.
        
        Sets:
            self.lattice_scaler: Scaler for lattice parameters
            self.scaler: Scaler for target property
        """
        # Load once to compute property scaler
        if scaler_path is None:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice')
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key=train_dataset.prop)
        else:
            self.lattice_scaler = torch.load(
                Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training/validation/testing.
        
        Instantiates datasets from configuration and assigns data scalers
        to each dataset. This method is called automatically by PyTorch Lightning.
        
        Args:
            stage: One of 'fit', 'test', or None (both). Determines which
                datasets to instantiate:
                - 'fit': Only train and validation datasets
                - 'test': Only test datasets
                - None: All datasets
        """
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
