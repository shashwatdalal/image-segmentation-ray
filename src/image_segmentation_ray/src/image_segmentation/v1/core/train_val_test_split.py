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

from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_column(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    split_col_name: str,
    split_labels: Dict[str, str],
) -> pd.DataFrame:
    """Generates a column differentiating train, validation and test sets(optional).

    Train and validation ratio are set as parameters and test ratio
    is derived from the first two in case test set is required
    (i.e. train ratio + validation ratio < 1).

    Args:
        df: table
        train_ratio: ratio of train records
        val_ratio: ratio of validation records
        split_col_name: split column name
        split_labels: train, validation and test labels
    Returns:
        df table with an additional column categorizing
        train, validation and test records.
    """
    x_train, x_val_test = train_test_split(df, train_size=train_ratio)
    x_train[split_col_name] = split_labels["train"]
    x_val, x_test = train_test_split(
        x_val_test, train_size=val_ratio / (1 - train_ratio)
    )
    x_val[split_col_name] = split_labels["validation"]
    x_test[split_col_name] = split_labels["test"]
    df_w_data_split = pd.concat([x_train, x_val, x_test], axis="rows")
    return df_w_data_split
