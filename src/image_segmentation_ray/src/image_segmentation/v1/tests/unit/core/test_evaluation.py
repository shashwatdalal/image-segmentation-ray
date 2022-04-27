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

from image_segmentation.v1.core.evaluation import (
    compute_dice_loss,
    compute_iou,
    evaluate_predictions,
)


# pylint: disable=missing-function-docstring
def test_compute_iou(true_mask, predicted_mask):
    expected_output = {"iou_0": 0.5, "iou_1": 5 / 7}
    generated_output = compute_iou(true_mask=true_mask, predicted_mask=predicted_mask)
    assert generated_output == expected_output


# pylint: disable=missing-function-docstring
def test_compute_dice_loss(true_mask, predicted_mask):
    expected_output = {"dice_0": 2 / 3, "dice_1": 5 / 6}
    generated_output = compute_dice_loss(
        true_mask=true_mask, predicted_mask=predicted_mask
    )
    assert generated_output == expected_output


# pylint: disable=missing-function-docstring
def test_evaluate(metadata_w_train_test_split):
    evaluation_metrics = [compute_iou, compute_dice_loss]
    generated_output = evaluate_predictions(
        metadata_w_train_test_split,
        evaluation_metrics=evaluation_metrics,
        splits_to_predict=["TEST"],
    )
    assert {"iou_0", "iou_1", "dice_0", "dice_1"}.issubset(
        set(generated_output.columns)
    )
