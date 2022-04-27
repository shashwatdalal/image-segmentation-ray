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

# pylint: skip-file
# flake8: noqa
"""Tests."""

import pandas as pd
import pytest

from refit.v1.core.remove_debug_columns import (
    remove_input_debug_columns,
    remove_output_debug_columns,
)

PREFIX = "_"


@pytest.fixture
def sample_el():
    """Sample elasticsearch response."""
    return [
        {
            "_index": "nhs_demo",
            "_type": "_doc",
            "_id": "dLqZPnAB79C17UGFXPlq",
            "_score": 23.907246,
            "_source": {
                "site_id": "4",
                "site_name": "#2180632 natl ry sedol #ca7 canadyan com isin co",
            },
        },
        {
            "_index": "nhs_demo",
            "_type": "_doc",
            "_id": "l7qZPnAB79C17UGFXPlr",
            "_score": 23.29108,
            "_source": {
                "site_id": "39",
                "site_name": "#ca7 com sedol co canadhan ry #2180632 natl isin",
            },
        },
    ]


@pytest.fixture
def sample_el_df_pd(sample_el):
    df = pd.DataFrame(sample_el)
    df["non_hidden"] = 1
    return df


@pytest.fixture
def sample_dict(sample_el_df_pd):
    dict1 = {}
    dict1["key"] = sample_el_df_pd
    return dict1


@pytest.fixture
def sample_el_df_spark(sample_el_df_pd, spark):
    return spark.createDataFrame(sample_el_df_pd)


@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func1(df1):
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    assert len(debug_cols) == 0
    return df1


@remove_input_debug_columns()
def some_func2(df1, df2, dict1):
    debug_cols1 = [x for x in df1.columns if x.startswith(PREFIX)]
    debug_cols2 = [x for x in df2.columns if x.startswith(PREFIX)]
    assert len(debug_cols1) == 0
    assert len(debug_cols2) == 0
    assert dict1["key"] == "value"
    assert "non_hidden" in df1.columns
    assert "non_hidden" in df2.columns


@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func3(df1):
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    df1["new_col1"] = 1
    df1["_new_col2"] = 2
    assert len(debug_cols) == 0
    return df1


def test_remove_debug_cols1_pd(sample_el_df_pd):
    some_func1(sample_el_df_pd)


def test_remove_debug_cols2_pd(sample_el_df_pd):
    some_func2(sample_el_df_pd, sample_el_df_pd, {"key": "value"})


def test_remove_debug_cols3_pd_kwargs(sample_dict):
    df1 = some_func3(df1=sample_dict["key"])
    debug_cols = [x for x in df1.columns if x.startswith(PREFIX)]
    assert len(debug_cols) == 0


def test_remove_debug_cols1_spark(sample_el_df_spark):
    some_func1(sample_el_df_spark)


def test_remove_debug_cols2_spark_kwargs(sample_el_df_spark):
    some_func2(df1=sample_el_df_spark, df2=sample_el_df_spark, dict1={"key": "value"})


def test_remove_output_debug_cols(sample_el_df_spark):
    df = some_func1(sample_el_df_spark)
    cols_start_prefix = [column for column in df.columns if column.startswith(PREFIX)]
    assert cols_start_prefix == []
