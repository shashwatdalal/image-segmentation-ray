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
"""Train validation test split."""

from refit.v1.core.augment import augment

from ..core import train_val_test_split


@augment()
def train_val_test_column(*args, **kwargs):
    """Augment wrapper around `train_val_test_column`.

    See core function for more details.
    """
    return train_val_test_split.train_val_test_column(*args, **kwargs)
