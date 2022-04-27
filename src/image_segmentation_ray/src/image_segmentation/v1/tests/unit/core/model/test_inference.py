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
"""Tests inference."""

import os

import skimage
import torch
from monai.networks.nets import UNet

from image_segmentation.v1.core.model.inference import predict_masks


# pylint: disable=missing-function-docstring
def test_predict_masks(metadata_w_train_test_split, split_col_name, input_path):
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=[4, 8, 16],
        strides=[2, 2],
        num_res_units=2,
    )
    post_process_functions = [
        (skimage.morphology.remove_small_holes, {"area_threshold": 2}),
        (skimage.morphology.remove_small_objects, {"min_size": 2}),
    ]

    predicted_mask_path = os.path.join(input_path, "ID_4/predicted_mask.png")
    # Remove predicted_mask.png file in case it already exists
    is_predicted_mask = os.path.isfile(predicted_mask_path)
    if is_predicted_mask:
        os.remove(predicted_mask_path)

    # pylint: disable=no-member
    dummy_boolean = predict_masks(
        metadata=metadata_w_train_test_split,
        split_col_name=split_col_name,
        splits_to_predict=["TEST"],
        model=model,
        threshold=0.7,
        output_layer_activ_function=torch.sigmoid,
        post_process_functions=post_process_functions,
        use_gpu=False,
    )
    assert dummy_boolean

    assert os.path.isfile(os.path.join(input_path, "ID_4/predicted_mask.png"))
