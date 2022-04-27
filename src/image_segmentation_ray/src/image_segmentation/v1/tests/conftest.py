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

"""Tests fixtures."""

import os
import pathlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(name="input_path")
def input_path_fixture():
    """Generate input path for tests."""
    return os.path.join(pathlib.Path(__file__).parent.resolve(), "input")


@pytest.fixture
def masks_list():
    """Generates a list of masks."""
    mask_1 = np.array([[3.2, 0, 0, 3.2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    mask_2 = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    mask_3 = np.array([[0, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]])
    return [mask_1, mask_2, mask_3]


@pytest.fixture
def metadata():
    """Generates metadata table."""
    return pd.DataFrame(
        {
            "id": ["ID_1", "ID_2", "ID_3", "ID_4"],
            "img_path": [
                "ID_1/image.png",
                "ID_2/image.png",
                "ID_3/image.png",
                "ID_4/image.png",
            ],
            "mask_path": [
                "ID_1/mask.png",
                "ID_2/mask.png",
                "ID_3/mask.png",
                "ID_4/mask.png",
            ],
        }
    )


@pytest.fixture
def metadata_w_train_test_split(input_path):
    """Generates metadata table."""
    return pd.DataFrame(
        {
            "id": ["ID_1", "ID_2", "ID_3", "ID_4"],
            "img_path": [
                input_path + "/ID_1/image.png",
                input_path + "/ID_2/image.png",
                input_path + "/ID_3/image.png",
                input_path + "/ID_4/image.png",
            ],
            "mask_path": [
                input_path + "/ID_1/mask.png",
                input_path + "/ID_2/mask.png",
                input_path + "/ID_3/mask.png",
                input_path + "/ID_4/mask.png",
            ],
            "split_col_name": ["TRAIN", "TRAIN", "VAL", "TEST"],
        }
    )


@pytest.fixture
def split_col_name():
    """Generates split name."""
    return "split_col_name"


@pytest.fixture
def split_labels():
    """Generates split labels."""
    return {"train": "TRAIN", "validation": "VAL", "test": "TEST"}


@pytest.fixture
def true_mask():
    """True mask example."""
    return np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]])


@pytest.fixture
def predicted_mask():
    """Predicted mask example."""
    return np.array([[0, 1, 1], [0, 1, 1], [1, 0, 1]])
