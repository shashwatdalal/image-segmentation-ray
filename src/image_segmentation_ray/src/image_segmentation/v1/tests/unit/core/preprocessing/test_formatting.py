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

"""Data formatting tests."""

import numpy as np
import pytest

from image_segmentation.v1.core.preprocessing.formatting import (
    combine_masks_into_single_mask,
)


# pylint: disable=missing-function-docstring
def test_combine_masks(masks_list):
    expected_output = np.array([[2, 2, 0, 1], [2, 3, 3, 0], [0, 3, 3, 0], [0, 0, 0, 0]])
    generated_output = combine_masks_into_single_mask(masks_list=masks_list)
    assert np.array_equal(generated_output, expected_output)


# pylint: disable=missing-function-docstring
def test_combine_masks_exception(masks_list):
    # Mask size in masks_list are 4x4, add a mask with a different size
    additional_mask = np.array([[2, 2, 0], [2, 3, 3], [0, 3, 3], [0, 0, 0]])
    masks_list.append(additional_mask)
    with pytest.raises(ValueError):
        combine_masks_into_single_mask(masks_list=masks_list)
