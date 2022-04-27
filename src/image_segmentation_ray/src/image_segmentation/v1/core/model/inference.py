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
"""Inference."""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import skimage
import torch
from torchvision.io import read_image

from src.image_segmentation_ray.src.image_segmentation.v1.core.postprocessing import (
    post_process_np_array,
)


def predict_masks(
    metadata: pd.DataFrame,
    model: torch.nn.Module,
    threshold: float,
    output_layer_activ_function: Callable,
    post_process_functions: List[Tuple[Callable, Dict]] = None,
    split_col_name: str = None,
    splits_to_predict: List[str] = None,
    use_gpu: bool = False,
) -> bool:
    """Predicts masks and then applies post-processing (if needed).

    The function writes the resulting masks in the locations specified
    in metadata table.

    Args:
        metadata: table containing images and mask paths and potentially
            other relevant information specific to the use case
        model: model to be used
        threshold: lower threshold to trigger pixel classification (between 0 and 1)
        output_layer_activ_function: output layer activation function
        post_process_functions: (Optional) list of tuples including functions
            (and their arguments) to apply to image (in array format). One can
            add pre-built functions (e.g. scipy.morphology.remove_small_objects
            or any other scipy.morphology function) or their own custom functions
        split_col_name: (Optional) split column name. If not provided, a
            prediction will be made for every image listed in the metadata.
        splits_to_predict: (Optional) list of split categories to predict.
            The default is to make a prediction for all images listed in the
            metadata.
        use_gpu: (Optional) boolean to indicate whether to use GPU for inference
            or not. If not provided, CPUs will be used.

    Returns:
        dummy boolean
    """
    if splits_to_predict is not None and split_col_name is not None:
        metadata = metadata[metadata[split_col_name].isin(splits_to_predict)]

    # Switch model to evaluation mode
    model.eval()

    for img_path in metadata["img_path"]:
        # Prediction
        image = read_image(img_path)
        image = image.unsqueeze(0)
        # pylint: disable=no-member
        if torch.cuda.is_available() and use_gpu:
            image = image.type(torch.cuda.FloatTensor)
            model.to("cuda")
            image.to("cuda")
            with torch.no_grad():
                predicted_mask = output_layer_activ_function(model(image))
        else:
            image = image.type(torch.FloatTensor)
            predicted_mask = output_layer_activ_function(model(image))
        predicted_mask = predicted_mask >= threshold
        numpy_predicted_mask = predicted_mask.cpu().detach().numpy()

        # Post-processing
        if post_process_np_array is not None:
            numpy_predicted_mask = post_process_np_array(
                numpy_predicted_mask, post_process_functions,
            )

        # Write
        skimage.io.imsave(
            img_path.rsplit("/", 1)[0] + "/predicted_mask.png",
            skimage.img_as_ubyte(np.squeeze(numpy_predicted_mask)),
        )
    return True
