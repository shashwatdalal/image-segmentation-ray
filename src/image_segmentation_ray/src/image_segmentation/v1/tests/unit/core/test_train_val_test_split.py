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

"""Tests train, validation and test split"""


from image_segmentation.v1.core.train_val_test_split import train_val_test_column


# pylint: disable=missing-function-docstring
def test_train_val_test_column(metadata, split_col_name, split_labels):
    generated_output = train_val_test_column(
        df=metadata,
        train_ratio=0.5,
        val_ratio=0.25,
        split_col_name=split_col_name,
        split_labels=split_labels,
    )
    assert (
        len(generated_output[generated_output[split_col_name] == split_labels["train"]])
        == 2
    )
    assert (
        len(
            generated_output[
                generated_output[split_col_name] == split_labels["validation"]
            ]
        )
        == 1
    )
    assert (
        len(generated_output[generated_output[split_col_name] == split_labels["test"]])
        == 1
    )
