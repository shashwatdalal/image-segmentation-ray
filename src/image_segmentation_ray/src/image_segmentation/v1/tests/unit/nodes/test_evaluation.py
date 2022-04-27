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

"""Tests evaluation functions."""

from image_segmentation.v1.nodes.evaluation import evaluate_predictions


# pylint: disable=missing-function-docstring
def test_evaluate_predictions(metadata_w_train_test_split):
    iou = "pmpx_pkg.utilities.image_segmentation.v1.core.evaluation.compute_iou"
    dice_loss = (
        "pmpx_pkg.utilities.image_segmentation.v1.core.evaluation.compute_dice_loss"
    )
    evaluation_metrics = [
        {"object": iou},
        {"object": dice_loss},
    ]
    generated_output = evaluate_predictions(
        metadata_w_train_test_split,
        evaluation_metrics=evaluation_metrics,
        splits_to_predict=["TEST"],
    )
    assert {"iou_0", "iou_1", "dice_0", "dice_1"}.issubset(
        set(generated_output.columns)
    )
