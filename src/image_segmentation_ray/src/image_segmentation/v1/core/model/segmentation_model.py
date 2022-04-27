# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""Segmentation model."""

from typing import Tuple

import pytorch_lightning as ptl
import torch
from torch import nn


# pylint: disable=abstract-method
# pylint: disable=too-many-ancestors
class SegmentationModel(ptl.LightningModule):
    """Segmentation model.

    This class contains the used segmentation model along with
    parameters relevant for model training, such as optimizer, loss function,
    learning rate.
    """

    def __init__(self, model: nn.Module, learning_rate: float, loss: nn.Module):
        """Ran once when instantiating the SegmentationModel object."""
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.train_loss = []

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step.

        Args:
            x: an image
        Returns:
            Predicted mask.
        """
        return self.model(x)

    # pylint: disable=arguments-differ
    # pylint: disable=unused-argument
    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> float:
        """Training step.

        Args:
            batch: tuple of images and masks
            batch_idx: batch index
        Returns:
            Loss value.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.float())
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.train_loss.append(loss)
        return loss

    # pylint: disable=arguments-differ
    # pylint: disable=unused-argument
    def validation_step(self, batch: Tuple[torch.Tensor], batch_idx: int):
        """Validation step.

        Args:
            batch: tuple of images and masks
            batch_idx: batch index.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.float())
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        """Optimizer configuration."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
