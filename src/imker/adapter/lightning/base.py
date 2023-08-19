import hashlib
import pickle
from typing import Any, Optional

import lightning.pytorch as pl
import torch

from ...inspection import hasfunc, parse_arguments
from ...types import ArrayLike


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        loss: type[torch.nn.Module],
        optimizer: type[torch.optim.Optimizer],
        lr_scheduler: type[torch.optim.lr_scheduler._LRScheduler],
        loss_params: Optional[dict[str, Any]] = None,
        optimizer_params: Optional[dict[str, Any]] = None,
        lr_scheduler_params: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.loss = loss(**loss_params) if loss_params is not None else loss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params

    def training_step(self, batch: dict, batch_idx: int):
        if hasfunc(self, "compute_loss"):
            loss = self.compute_loss(batch)
        else:
            loss = self.default_compute_loss(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if hasfunc(self, "compute_loss"):
            loss = self.compute_loss(batch)
        else:
            loss = self.default_compute_loss(batch)
        return {"loss": loss.detach()}

    def default_compute_loss(self, batch: dict):
        y = batch.pop("y")
        X = batch.pop("X")
        out = self.forward(X)
        loss = self.loss(out, y)
        return loss

    def training_epoch_end(self, outputs_train):
        # OPTIONAL
        loss = torch.stack([val["loss"] for val in outputs_train]).mean()
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
        )

        print(f"Epoch[{self.current_epoch}] train loss: {loss}")

    def validation_epoch_end(self, outputs_valid):
        # OPTIONAL
        loss = torch.stack([val["loss"] for val in outputs_valid]).mean()
        self.log(
            "valid_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
        )
        print(f"Epoch[{self.current_epoch}] valid loss: {loss}")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, X: ArrayLike, y: Optional[ArrayLike] = None, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        _args = parse_arguments(self.__init__)
        if "X" not in _args and "y" not in _args:
            raise AttributeError("__init__() must accept both X and y as arguments.")
        else:
            return self


class BaseLightningTask(object):
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eval_set: Optional[list[tuple[ArrayLike, ArrayLike]]] = None,
    ):
        return self

    def get_identifier(self, X: ArrayLike) -> str:
        return hashlib.sha256(pickle.dumps(X)).hexdigest()

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if hasfunc(cls, ("transform", "split", "predict", "predict_proba"), hasany=True):
            return self
        else:
            raise NotImplementedError(
                "Task hasn't any required method, you should implement one of the transform(), \
split(), predict() or predict_proba()"
            )
