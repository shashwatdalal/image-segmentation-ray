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
"""Trainer."""

import pandas as pd
import pytorch_lightning as ptl
import torch
import torchvision


# pylint: disable=too-many-arguments
from src.image_segmentation_ray.src.image_segmentation.v1.core.model.dataset import (
    ImageDataset,
)
from src.image_segmentation_ray.src.image_segmentation.v1.core.model.segmentation_model import (
    SegmentationModel,
)


def train(
    metadata: pd.DataFrame,
    split_col_name: str,
    split_labels: dict,
    data_loader_kwargs: dict,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    learning_rate: float,
    trainer_kwargs: dict,
    transforms: torchvision.transforms.Compose = None,
    target_transforms: torchvision.transforms.Compose = None,
) -> ptl.Trainer:
    """Train model.

    Args:
        metadata: table containing images and mask paths and potentially
            other relevant information specific to the use case
        split_col_name: split split column name
        split_labels: train, validation and test labels
        data_loader_kwargs: torch.utils.data.DataLoader kwargs
            See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        model: model to be used
        loss: loss function
        learning_rate: Adjustment in weights of our neural network
            with respect to the loss gradient descent
        trainer_kwargs: The pytorch_lightening.trainer kwargs
            See https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
        transforms: transforms that need to be applied to training images
        target_transforms: transforms that need to be applied to masks

    Return:
        Trained model
    """
    # Create PyTorch Dataset
    train_dataset = ImageDataset(
        metadata[metadata[split_col_name] == split_labels["train"]],
        transforms=transforms,
        target_transforms=target_transforms,
    )
    val_dataset = ImageDataset(
        metadata[metadata[split_col_name] == split_labels["validation"]],
        transforms=transforms,
        target_transforms=target_transforms,
    )

    # Using the dataset, create the data loader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, **data_loader_kwargs
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, **data_loader_kwargs
    )

    # Create the model
    model = SegmentationModel(model=model, learning_rate=learning_rate, loss=loss)
    trainer = ptl.Trainer(**trainer_kwargs)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return trainer.model
