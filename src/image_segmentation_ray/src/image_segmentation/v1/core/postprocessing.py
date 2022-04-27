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
"""Post-processing."""

from typing import Callable, Dict, List, Tuple

import numpy as np


def post_process_np_array(
    x: np.ndarray, post_process_functions: List[Tuple[Callable, Dict]],
) -> np.ndarray:
    """Applies a sequence of functions to a numpy array.

    For example if the array is an image with segments and there are small
    holes in some segments you can add scipy.morphology.remove_small_holes
    in the list of functions to apply.

    Args:
        x: array to apply the sequence of transformations to
        post_process_functions: list of tuples including functions
            (and their arguments) to apply to array `x`. One can add already
            built functions (e.g. scipy.morphology.remove_small_objects or any
            other scipy.morphology function) or add implemented functions
    Returns:
        Transformed array.
    """
    for (func, kwargs) in post_process_functions:
        x = func(x, **kwargs)
    return x
