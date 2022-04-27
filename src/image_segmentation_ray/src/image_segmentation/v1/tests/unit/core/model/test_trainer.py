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
"""Tests trainer function."""

import torchvision
from monai.networks.nets import UNet
from torch.nn import BCEWithLogitsLoss

from image_segmentation.v1.core.model.trainer import train


# pylint: disable=missing-function-docstring
def test_train(metadata_w_train_test_split, split_col_name, split_labels):
    data_loader_kwargs = {"batch_size": 2, "shuffle": True}
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=[4, 8, 16],
        strides=[2, 2],
        num_res_units=2,
    )
    loss = BCEWithLogitsLoss()
    trainer_kwargs = {"max_epochs": 6, "fast_dev_run": False}
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop(size=256)]
    )
    target_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop(size=256)]
    )
    model = train(
        metadata=metadata_w_train_test_split,
        split_col_name=split_col_name,
        split_labels=split_labels,
        data_loader_kwargs=data_loader_kwargs,
        model=model,
        loss=loss,
        learning_rate=1e-2,
        trainer_kwargs=trainer_kwargs,
        transforms=transforms,
        target_transforms=target_transforms,
    )
    assert model.train_loss[0] != model.train_loss[-1]
