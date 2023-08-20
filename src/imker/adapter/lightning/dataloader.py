import random
from typing import Any, Callable, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

from ...types import ArrayLike
from .base import BaseDataset


def seed_worker(worker_seed: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class LightningDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.seed = seed

    def _train_dataloader(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        dataset: type[BaseDataset],
        dataset_params: Optional[dict[str, Any]] = None,
    ) -> DataLoader:
        if dataset_params is None:
            _dataset = dataset(X=X_train, y=y_train)
        else:
            _dataset = dataset(X=X_train, y=y_train, **dataset_params)

        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            _dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def _valid_dataloader(
        self,
        X_valid: ArrayLike,
        y_valid: ArrayLike,
        dataset: type[BaseDataset],
        dataset_params: Optional[dict[str, Any]] = None,
    ) -> DataLoader:
        if dataset_params is None:
            _dataset = dataset(X=X_valid, y=y_valid)
        else:
            _dataset = dataset(X=X_valid, y=y_valid, **dataset_params)
        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            _dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def _test_dataloader(
        self,
        X_test: ArrayLike,
        dataset: type[BaseDataset],
        dataset_params: Optional[dict[str, Any]] = None,
    ) -> DataLoader:
        if dataset_params is None:
            _dataset = dataset(X=X_test)
        else:
            _dataset = dataset(X=X_test, **dataset_params)

        return DataLoader(
            _dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
