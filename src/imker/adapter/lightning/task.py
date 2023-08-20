import copy
import gc
import glob
import os
from typing import Any, Callable, Iterable, Literal, Optional, Union

import torch

import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
from lightning.pytorch import seed_everything
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import Strategy

from ...inspection import get_code
from ...types import ArrayLike
from .base import BaseDataset, BaseLightningModule, BaseLightningTask
from .dataloader import LightningDataLoader

_PRECISION_TYPE = Union[
    Literal[64, 32, 16],
    Literal["16-mixed", "bf16-mixed", "32-true", "64-true"],
    Literal["64", "32", "16", "bf16"],
]

_LOGGER_TYPE = Optional[Union[Logger, Iterable[Logger], bool]]


class LightningTask(BaseLightningTask):
    def __init__(
        self,
        model: type[BaseLightningModule],
        train_dataset: type[BaseDataset],
        test_dataset: type[BaseDataset],
        valid_dataset: Optional[type[BaseDataset]] = None,
        model_init_params: Optional[dict[str, Any]] = None,
        train_dataset_params: Optional[dict[str, Any]] = None,
        valid_dataset_params: Optional[dict[str, Any]] = None,
        test_dataset_params: Optional[dict[str, Any]] = None,
        epochs: int = 30,
        early_stopping_round: Optional[int] = None,
        min_delta: float = 1e-5,
        monitor: str = "valid_loss",
        collate_fn: Optional[Callable] = None,
        precision: _PRECISION_TYPE = "16-mixed",
        limit_train_batches: float = 1.0,
        move_metrics_to_cpu: bool = False,
        plugins: Any = None,
        loader_num_workers: int = 0,
        pin_memory: bool = False,
        ckpt_name: str = "best_model",
        batch_size: int = 16,
        checkpoint_path: str = "./checkpoint",
        seed: int = 42,
        save_every_n_epochs: Optional[int] = None,
        save_top_k: int = 1,
        accelerator: Union[str, Accelerator] = "auto",
        devices: Union[list[int], str, int] = "auto",
        strategy: Union[str, Strategy] = "auto",
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[float] = None,
        sync_batchnorm: bool = False,
        logger: Optional[_LOGGER_TYPE] = None,
    ):
        self.model_class = model
        self.model_init_params = (
            copy.deepcopy(model_init_params) if model_init_params is not None else dict()
        )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.train_dataset_params = (
            train_dataset_params if train_dataset_params is not None else dict()
        )
        self.valid_dataset_params = (
            valid_dataset_params if valid_dataset_params is not None else dict()
        )
        self.test_dataset_params = (
            test_dataset_params if test_dataset_params is not None else dict()
        )
        self.epochs = epochs
        self.early_stopping_round = (
            early_stopping_round if early_stopping_round is not None else epochs
        )
        self.min_delta = min_delta
        self.monitor = monitor
        self.collate_fn = collate_fn
        self.precision = precision
        self.limit_train_batches = limit_train_batches
        self.move_metrics_to_cpu = move_metrics_to_cpu
        self.plugins = plugins
        self.loader_num_workers = loader_num_workers
        self.pin_memory = pin_memory
        self.ckpt_name = ckpt_name
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.save_every_n_epochs = save_every_n_epochs
        self.save_top_k = save_top_k
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.sync_batchnorm = sync_batchnorm
        self.logger = logger

        os.makedirs(f"{self.checkpoint_path}", exist_ok=True)
        self.model_id = len(glob.glob(f"{self.checkpoint_path}/{self.ckpt_name}*.ckpt"))

    def _get_default_callbacks(self) -> list:
        callbacks = [
            plc.EarlyStopping(
                monitor=self.monitor,
                min_delta=self.min_delta,
                patience=self.early_stopping_round,
                mode="min",
            ),
            plc.ModelCheckpoint(
                monitor=self.monitor,
                dirpath=f"{self.checkpoint_path}",
                filename=f"{self.ckpt_name}_{self.model_id}",
                mode="min",
                save_weights_only=True,
                every_n_epochs=self.save_every_n_epochs,
                save_top_k=self.save_top_k,
            ),
        ]

        return callbacks

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eval_set: Optional[list[tuple[ArrayLike, ArrayLike]]] = None,
        callbacks: Optional[Union[list[Callback], Callback]] = None,
    ):
        if callbacks is None:
            callbacks = self._get_default_callbacks()

        seed_everything(self.seed, workers=True)
        model = self.model_class(**self.model_init_params)

        dataloader = LightningDataLoader(
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            seed=self.seed,
        )

        train_dataloader = dataloader._train_dataloader(
            X,
            y,
            dataset=self.train_dataset,
            dataset_params=self.train_dataset_params,
        )

        if eval_set is None:
            valid_dataloader = None
        else:
            X_valid, y_valid = eval_set[0]
            if isinstance(self.valid_dataset, type):
                valid_dataloader = dataloader._valid_dataloader(
                    X_valid,
                    y_valid,
                    dataset=self.valid_dataset,
                    dataset_params=self.valid_dataset_params,
                )
            else:
                raise

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            precision=self.precision,
            default_root_dir=self.checkpoint_path,
            logger=self.logger,
            limit_train_batches=self.limit_train_batches,
            # move_metrics_to_cpu=self.move_metrics_to_cpu,
            plugins=self.plugins,
            deterministic=True,
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=self.gradient_clip_val,
            sync_batchnorm=self.sync_batchnorm,
        )

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        if os.path.exists(f"{self.checkpoint_path}/{self.ckpt_name}_{self.model_id}.tmp_end.ckpt"):
            os.remove(f"{self.checkpoint_path}/{self.ckpt_name}_{self.model_id}.tmp_end.ckpt")

        del trainer
        del train_dataloader
        del valid_dataloader
        del dataloader

        gc.collect()

        return self

    def forward(
        self,
        X: ArrayLike,
        checkpoint: Optional[str] = None,
        accelerator: Union[str, Accelerator] = "auto",
        devices: Union[list[int], str, int] = "auto",
        strategy: Union[str, Strategy] = "auto",
        model: Optional[BaseLightningModule] = None,
    ):
        if checkpoint is None:
            checkpoint = f"{self.checkpoint_path}/{self.ckpt_name}_{self.model_id}.ckpt"

        if model is None:
            model = self.model_class.load_from_checkpoint(
                checkpoint_path=checkpoint,
            )

        model.eval()
        model.freeze()

        dataloader = LightningDataLoader(
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

        test_dataloader = dataloader._test_dataloader(
            X,
            dataset=self.test_dataset,
            dataset_params=self.test_dataset_params,
        )

        trainer = pl.Trainer(
            max_epochs=-1,
            logger=False,
            enable_checkpointing=False,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            enable_model_summary=False,
        )

        pred = trainer.predict(model=model, dataloaders=test_dataloader)
        if pred is not None:
            pred = torch.cat([p.cpu().detach() for p in pred]).numpy()  # type: ignore

        gc.collect()
        return pred

    def get_code(self):
        return get_code(self.model_class)
