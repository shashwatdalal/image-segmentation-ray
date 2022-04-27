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
"""Evaluation."""
import logging
import os
from typing import Callable, Generator, List

import numpy as np
import pandas as pd
import skimage
from sklearn import metrics

from src.image_segmentation_ray.src.image_segmentation.v1.core.internals import (
    CONSTANTS,
)


def evaluate_predictions(
    metadata: pd.DataFrame,
    evaluation_metrics: List[Callable],
    splits_to_predict: List[str] = None,
) -> pd.DataFrame:
    """Evaluates the model using image segmentation metrics.

    Args:
        metadata: table containing image and mask paths and potentially
            other relevant information specific to the use case
        evaluation_metrics: list of evaluation metrics to apply. Each
            evaluation metric has to be a function with input arguments "true_mask"
            and "predicted_mask", each taking in a numpy array, and returns a
            dictionary where the keys are the metric names. One can either have 1
            metric value per image or multiple ones per image
            i.e. one for each class.
        splits_to_predict: (Optional) list of split categories to predict. If not
            provided, all images listed in metadata get a predicted mask from the
            inference and they all get evaluated using this function.

    Returns:
        Metadata with evaluation metrics values.
    """
    if splits_to_predict is not None:
        metadata = metadata[metadata[CONSTANTS.SPLIT_COL_NAME].isin(splits_to_predict)]

    metrics_values = []
    for true_mask, predicted_mask, mask_path in read_images(metadata):
        if true_mask.shape != predicted_mask.shape:
            msg = (
                "Please make sure true and predicted masks have the same shape.",
                f"The mask in {mask_path} has been skipped.",
            )
            logging.warning("\n".join(msg))
            continue

        prediction_perf = {}
        for evaluation_metric in evaluation_metrics:
            value = evaluation_metric(
                true_mask=true_mask, predicted_mask=predicted_mask
            )
            prediction_perf.update(value)
        prediction_perf[CONSTANTS.TRUE_MASK_PATH_COL_NAME] = mask_path
        metrics_values.append(prediction_perf)
    metadata = pd.merge(
        metadata,
        pd.DataFrame(metrics_values),
        how="left",
        on=CONSTANTS.TRUE_MASK_PATH_COL_NAME,
    )
    return metadata


def compute_iou(true_mask: np.ndarray, predicted_mask: np.ndarray) -> dict:
    """Computes IoU (intersection over union).

    IoU is the ratio:
    |intersection(true_mask, predicted_mask)| /
    |union(true_mask, predicted_mask)|. This is computed for every class:
    For example, for class 0, |intersection(true_mask, predicted_mask)|
    is the number of pixels that have a 0-value both in true_mask and
    predicted_mask. Whereas |union(true_mask, predicted_mask)| is the
    total number of 0-value pixels in union(true_mask, predicted_mask).

    Args:
        true_mask: ground truth
        predicted_mask: predicted mask
    Returns:
        Array of IoUs for each class
    """
    confusion_matrix = metrics.confusion_matrix(
        true_mask.flatten(), predicted_mask.flatten()
    )
    iou = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1)
        + np.sum(confusion_matrix, axis=0)
        - np.diag(confusion_matrix)
    )
    return {f"iou_{i}": value for i, value in enumerate(iou)}


def compute_dice_loss(true_mask: np.ndarray, predicted_mask: np.ndarray) -> dict:
    """Computes Dice Loss.

    The Dice loss is the ratio:
    2 * |intersection(true_mask, predicted_mask)| /
    (|true_mask| + |predicted_mask|). This is computed for every class:
    For example, for class 0, |intersection(true_mask, predicted_mask)|
    is the number of pixels that have a 0-value both in true_mask and
    predicted_mask. Whereas |true_mask| + |predicted_mask| is the
    sum of 0-value pixels in true_mask plus the ones in predicted_mask.
    Here, pixels that have a 0-value in both true_mask and predicted_mask
    are double counted.

    Args:
        true_mask: ground truth
        predicted_mask: predicted mask
    Returns:
        Array of Dice coefficients for each class
    """
    confusion_matrix = metrics.confusion_matrix(
        true_mask.flatten(), predicted_mask.flatten()
    )
    dice = (
        2.0
        * np.diag(confusion_matrix)
        / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0))
    )
    return {f"dice_{i}": value for i, value in enumerate(dice)}


def read_images(metadata: pd.DataFrame) -> Generator:
    """Generator function reading images.

    It reads an image, and yields it instead of returning it.
    This avoids memory errors.

    Args:
        metadata: table containing images and mask paths and potentially
            other relevant information specific to the use case.

    Yields:
        True mask in numpy array format
        Predicted mask in numpy array format
        True mask path
    """
    for mask_path in metadata[CONSTANTS.TRUE_MASK_PATH_COL_NAME]:
        true_mask = skimage.io.imread(mask_path)
        predicted_mask = skimage.io.imread(
            os.path.join(mask_path.rsplit("/", 1)[0], CONSTANTS.PREDICTED_MASK_FILENAME)
        )
        yield true_mask, predicted_mask, mask_path
