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
from src.image_segmentation_ray.src.refit.v1.core.augment import augment
from ...core.model import inference


@augment()
def predict_masks(*args, **kwargs):
    """Augment wrapper around `predict_masks`.

    See core function for more details.
    """
    return inference.predict_masks(*args, **kwargs)
