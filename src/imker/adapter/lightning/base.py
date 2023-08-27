import copy
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

import lightning.pytorch as pl

from ...inspection import get_code, hasfunc, parse_arguments
from ...types import ArrayLike


class BaseLightningModule(ABC, pl.LightningModule):
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
        self.__training_step_outputs: list = []
        self.__validation_step_outputs: list = []
        # self.save_hyperparameters()

    def training_step(self, batch: dict, batch_idx: int) -> Any:
        if hasfunc(self, "compute_loss"):
            loss = self.compute_loss(batch)  # type: ignore
        else:
            loss = self.default_compute_loss(batch)

        self.log(
            "train_loss_step", loss.detach().cpu(), prog_bar=True, on_step=True, on_epoch=False
        )
        self.__training_step_outputs.append(loss.detach())
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Any:
        if hasfunc(self, "compute_loss"):
            loss = self.compute_loss(batch)  # type: ignore
        else:
            loss = self.default_compute_loss(batch)

        self.log(
            "valid_loss_step", loss.detach().cpu(), prog_bar=True, on_step=True, on_epoch=False
        )
        self.__validation_step_outputs.append(loss.detach())
        return loss

    def default_compute_loss(self, batch: dict) -> Any:
        y = batch.pop("y")
        X = batch.pop("X")
        out = self.forward(X)
        loss = self.loss(out, y)
        return loss

    def on_train_epoch_end(self) -> None:
        # OPTIONAL
        loss = torch.stack(self.__training_step_outputs).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )

        # print(f"Epoch[{self.current_epoch}] train loss: {loss}")
        self.__training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        # OPTIONAL
        loss = torch.stack(self.__validation_step_outputs).mean()
        self.log(
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        # print(f"Epoch[{self.current_epoch}] valid loss: {loss}")
        self.__validation_step_outputs.clear()

    @abstractmethod
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }

    def get_code(self):
        return get_code(self.__class__)


class BaseDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, X: ArrayLike, y: Optional[ArrayLike] = None, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
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

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        if hasfunc(cls, ("forward"), hasany=True):
            return self
        else:
            raise NotImplementedError(
                "Task hasn't any required method, you should implement forward()"
            )

    def get_state(self) -> dict[str, Any]:
        state = copy.deepcopy(self.__dict__)
        if "model_id" in state:
            del state["model_id"]
        return state
