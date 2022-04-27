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
"""Tests output filter."""
import pytest

from ..core.output_primary_key import add_output_primary_key


@add_output_primary_key()
def my_node_func_input(df):
    return df


# ----- primary key check for spark output dataframe without null check-----
def test_add_output_primary_key_spark_without_null_without_dupe(spark_df,):
    output_primary_key = {"columns": ["int_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_without_null_with_dupe(spark_df,):
    output_primary_key = {"columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns \['float_col'\] has either duplicate values or null values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)


def test_add_output_primary_key_spark_without_null_without_dupe_list1(spark_df,):
    output_primary_key = {"columns": ["int_col", "float_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_without_null_without_dupe_list2(spark_df,):
    output_primary_key = {"columns": ["int_col", "string_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns \['int_col', 'string_col'\] has either duplicate values or null values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)


# ----- primary key check for spark output dataframe with null check-----
def test_add_output_primary_key_spark_with_null_without_dupe(spark_df,):
    output_primary_key = {"nullable": True, "columns": ["int_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_with_null_without_dupe_with_list(spark_df,):
    output_primary_key = {"nullable": True, "columns": ["int_col", "string_col"]}

    result = my_node_func_input(spark_df, output_primary_key=output_primary_key)

    assert result.count() == spark_df.count()


def test_add_output_primary_key_spark_with_null_with_dupe(spark_df,):
    output_primary_key = {"nullable": True, "columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns \['float_col'\] has either duplicate values or null values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)


def test_add_output_primary_key_spark_with_null_val(spark_df,):
    output_primary_key = {"columns": ["string_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns \['string_col'\] has either duplicate values or null values.",
    ):
        my_node_func_input(spark_df, output_primary_key=output_primary_key)


# ----- primary key check for Pandas output dataframe without null check-----
def test_add_output_primary_key_pandas_without_null(pandas_df):
    output_primary_key = {"columns": ["int_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


def test_add_output_primary_key_pandas_without_null_with_dupe(pandas_df,):
    output_primary_key = {"columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns \['float_col'\] has either duplicate values or null values.",
    ):
        my_node_func_input(pandas_df, output_primary_key=output_primary_key)


def test_add_output_primary_key_pandas_without_null_with_list(pandas_df,):
    output_primary_key = {"columns": ["int_col", "float_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


# ----- primary key check for Pandas output dataframe with null check-----
def test_add_output_primary_key_pandas_with_null(pandas_df):
    output_primary_key = {"nullable": True, "columns": ["int_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


def test_add_output_primary_key_pandas_with_null_with_list(pandas_df,):
    output_primary_key = {"nullable": True, "columns": ["int_col", "float_col"]}

    result = my_node_func_input(pandas_df, output_primary_key=output_primary_key)

    assert result.shape == pandas_df.shape


def test_add_output_primary_key_pandas_with_null_with_dupe(pandas_df,):
    output_primary_key = {"nullable": True, "columns": ["float_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns \['float_col'\] has either duplicate values or null values.",
    ):
        my_node_func_input(pandas_df, output_primary_key=output_primary_key)


def test_add_output_primary_key_pandas_with_null_val(pandas_df):
    output_primary_key = {"nullable": True, "columns": ["string_col"]}

    with pytest.raises(
        TypeError,
        match="Primary key columns \['string_col'\] has either duplicate values or null values.",
    ):
        my_node_func_input(pandas_df, output_primary_key=output_primary_key)
