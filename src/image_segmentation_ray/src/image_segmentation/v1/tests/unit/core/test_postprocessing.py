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

"""Tests post-processing functions"""

import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects

from image_segmentation.v1.core.postprocessing import post_process_np_array


# pylint: disable=missing-function-docstring
def test_post_process_np_array():
    x = np.array([[0, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1]])
    expected_output = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
    post_process_functions = [
        (remove_small_holes, {"area_threshold": 2}),
        (remove_small_objects, {"min_size": 2}),
    ]
    generated_output = post_process_np_array(
        x=x, post_process_functions=post_process_functions
    )
    assert np.array_equal(generated_output, expected_output)
