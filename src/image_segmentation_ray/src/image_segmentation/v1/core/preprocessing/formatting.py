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
"""Data formatting."""

from typing import List

import numpy as np


def combine_masks_into_single_mask(masks_list: List[np.ndarray],) -> np.ndarray:
    """This function combines multiple masks into a single mask.

    Masks are provided as a list of numpy arrays where areas of interest
    (called segments) are marked with value > 0 whereas the background has value = 0.
    The masks are combined into a single mask that has a positive value
    if and only if one of the input masks had a segment in the pixel.
    The positive value corresponds to the list position of the last image in the list,
    that has a segment in the pixel. If there's an overlap, the maximum
    (and therefore latest) value will be kept. See below for an example.
    Each pixel of a mask corresponds to the same pixel in the image to be segmented.
    Therefore, all masks and image should have the same size.
    For example :

    Mask 1 : |1|1|0|0|
             |1|1|0|0|
             |0|0|0|0|
             |0|0|0|0|

    Mask 2 : |0|0|0|0|
             |0|2|2|0|
             |0|2|2|0|
             |0|0|0|0|

    Combined mask : |1|1|0|0|
                    |1|2|2|0|
                    |0|2|2|0|
                    |0|0|0|0|

    Args:
        masks_list: the list of masks, which will
         be combined into a single mask.

    Returns:
        mask: a single mask combining all masks' segments

    Raises:
        ValueError: if mask sizes are different.
    """
    # Combine the masks, using the first mask as reference mask
    ref_size = masks_list[0].shape
    output_mask = np.zeros(ref_size)
    for i, mask in enumerate(masks_list):
        if mask.shape != ref_size:
            raise ValueError(
                "Masks sizes are different, please make sure they are the same."
            )

        # Binarise mask
        mask = mask > 0

        # Assign an incrementing label to each mask
        mask = np.multiply(mask, i + 1)

        # Combine masks by taking the maximum values
        # This also implies, in case of overlap, that the greatest mask label are kept
        output_mask = np.maximum(output_mask, mask)

    return output_mask
