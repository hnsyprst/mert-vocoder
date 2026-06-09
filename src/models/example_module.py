from typing import cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
from torchmetrics import MeanMetric


class ExampleLightningModule(LightningModule):
    """
    Example Lightning module. Replace this with your own Lightning module.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: type[torch.optim.Optimizer],
        scheduler: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
    ):
        """
        Initialise the Lightning module.

        :param net: Model object to train and/or inference.
        :type net: torch.nn.Module
        :param optimizer: Optimizer class.
        :type optimizer: Type[torch.optim.Optimizer]
        :param scheduler: Optional learning rate scheduler class.
        :type scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]]
        """

        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = cast(torch.nn.Module, torch.compile(net))
        self.optimizer_class = optimizer
        self.scheduler_class = scheduler
        self.criterion = criterion
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def model_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of targets.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """

        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def training_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data containing the inputs and targets.
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :return: A tensor of losses between model predictions and targets.
        :rtype: torch.Tensor
        """

        loss, _preds, _targets = self.model_step(batch)
        self.train_loss(loss.detach())  # Just for logging, so detach before accumulating
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ):
        """
        Perform a single validataion step on a batch of data from the validation set.

        :param batch: A batch of data containing the inputs and targets.
        :param batch_idx: The index of the current batch.
        :type batch_idx: int
        :return: A tensor of losses between model predictions and targets.
        :rtype: torch.Tensor
        """

        loss, _preds, _targets = self.model_step(batch)
        self.valid_loss(loss.detach())  # Just for logging, so detach before accumulating
        self.log("valid/loss", self.valid_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(
        self,
    ) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        optimizer = self.optimizer_class(params=self.net.parameters())  # ty:ignore[missing-argument]
        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
