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
import pytest

from ..core.has_schema import has_schema


@has_schema()
def node_func(df):
    return df


@has_schema()
def node_func_full_example_multiple(df1, df2):
    return df1, df2


class TestHasSchemaSpark:
    def test_default_allow_subset_false(self, sample_df_spark_all_dtypes):
        with pytest.raises(
            AssertionError, match=r"Actual vs expected columns do not match up."
        ):
            _input_has_schema = {
                "df": "df",
                "expected_schema": {"site_id": "int", "site_name": "string"},
                "allow_subset": False,
            }
            _output_has_schema = {
                "expected_schema": {"site_id": "int", "site_name": "string"},
                "allow_subset": False,
            }
            node_func(
                df=sample_df_spark_all_dtypes,
                input_has_schema=_input_has_schema,
                output_has_schema=_output_has_schema,
            )

    def test_default_allow_subset_true(self, dummy_spark_df):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {"c1": "int"},
            "allow_subset": True,
        }
        _output_has_schema = {
            "expected_schema": {"c1": "int"},
            "allow_subset": True,
        }
        node_func(
            df=dummy_spark_df,
            input_has_schema=_input_has_schema,
            output_has_schema=_output_has_schema,
        )

    def test_full_types(self, sample_df_spark_all_dtypes):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {
                "int_col": "int",
                "long_col": "bigint",
                "string_col": "string",
                "float_col": "float",
                "double_col": "double",
                "date_col": "date",
                "datetime_col": "timestamp",
                "array_int": "array<int>",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": False,
        }
        _output_has_schema = {
            "expected_schema": {
                "int_col": "int",
                "long_col": "bigint",
                "string_col": "string",
                "float_col": "float",
                "double_col": "double",
                "date_col": "date",
                "datetime_col": "timestamp",
                "array_int": "array<int>",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": False,
        }
        node_func(
            df=sample_df_spark_all_dtypes,
            input_has_schema=_input_has_schema,
            output_has_schema=_output_has_schema,
        )

    def test_relax_true(self, sample_df_spark_all_dtypes):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {
                "int_col": "long",
                "long_col": "bigint",
                "string_col": "string",
                "float_col": "double",
                "double_col": "double",
                "date_col": "date",
                "datetime_col": "timestamp",
                "array_int": "array<numeric>",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": True,
        }
        _output_has_schema = {
            "expected_schema": {
                "int_col": "long",
                "long_col": "bigint",
                "string_col": "string",
                "float_col": "double",
                "double_col": "double",
                "date_col": "date",
                "datetime_col": "timestamp",
                "array_int": "array<numeric>",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": True,
        }
        node_func(
            df=sample_df_spark_all_dtypes,
            input_has_schema=_input_has_schema,
            output_has_schema=_output_has_schema,
        )

    def test_relax_false(self, sample_df_spark_all_dtypes):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {
                "int_col": "long",
                "long_col": "bigint",
                "string_col": "string",
                "float_col": "double",
                "double_col": "double",
                "date_col": "date",
                "datetime_col": "timestamp",
                "array_int": "array<numeric>",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": False,
        }
        _output_has_schema = {
            "expected_schema": {
                "int_col": "long",
                "long_col": "bigint",
                "string_col": "string",
                "float_col": "double",
                "double_col": "double",
                "date_col": "date",
                "datetime_col": "timestamp",
                "array_int": "array<numeric>",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": False,
        }
        with pytest.raises(ValueError, match=r"Diffs displayed only"):
            node_func(
                df=sample_df_spark_all_dtypes,
                input_has_schema=_input_has_schema,
                output_has_schema=_output_has_schema,
            )

    def test_full_example_multiple_inputs_outputs(self, sample_df_spark_all_dtypes):
        def _ip_has_schema(df):
            return {
                "df": df,
                "expected_schema": {
                    "int_col": "int",
                    "long_col": "bigint",
                    "string_col": "string",
                    "float_col": "float",
                    "double_col": "double",
                    "date_col": "date",
                    "array_int": "array<int>",
                    "datetime_col": "timestamp",
                },
                "allow_subset": True,
                "raise_exc": True,
                "relax": False,
            }

        def _op_has_schema(x):
            return {
                "expected_schema": {
                    "int_col": "int",
                    "long_col": "bigint",
                    "string_col": "string",
                    "float_col": "float",
                    "double_col": "double",
                    "date_col": "date",
                    "array_int": "array<int>",
                    "datetime_col": "timestamp",
                },
                "allow_subset": True,
                "raise_exc": True,
                "relax": False,
                "output": x,
            }

        _input_has_schema = [_ip_has_schema("df1"), _ip_has_schema("df2")]
        _output_has_schema = [_op_has_schema(0), _op_has_schema(1)]
        node_func_full_example_multiple(
            df1=sample_df_spark_all_dtypes,
            df2=sample_df_spark_all_dtypes,
            input_has_schema=_input_has_schema,
            output_has_schema=_output_has_schema,
        )


class TestHasSchemaPandas:
    def test_default_allow_subset_false(self, sample_df_pd_all_dtypes):
        with pytest.raises(
            AssertionError, match=r"Actual vs expected columns do not match up."
        ):
            _input_has_schema = {
                "df": "df",
                "expected_schema": {"site_id": "int", "site_name": "string"},
                "allow_subset": False,
            }
            _output_has_schema = {
                "expected_schema": {"site_id": "int", "site_name": "string"},
                "allow_subset": False,
            }
            node_func(
                df=sample_df_pd_all_dtypes,
                input_has_schema=_input_has_schema,
                output_has_schema=_output_has_schema,
            )

    def test_default_allow_subset_true(self, dummy_pd_df):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {"c1": "int64"},
            "allow_subset": True,
        }
        _output_has_schema = {
            "expected_schema": {"c1": "int64"},
            "allow_subset": True,
        }
        node_func(
            df=dummy_pd_df,
            input_has_schema=_input_has_schema,
            output_has_schema=_output_has_schema,
        )

    def test_full_types(self, sample_df_pd_all_dtypes):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {
                "float_col": "float64",
                "int_col": "int64",
                "datetime_col": "datetime64[ns]",
                "date_col": "object",
                "string_col": "object",
                "datetime_ms_col": "datetime64[ns]",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": False,
        }
        _output_has_schema = {
            "expected_schema": {
                "float_col": "float64",
                "int_col": "int64",
                "datetime_col": "datetime64[ns]",
                "date_col": "object",
                "string_col": "object",
                "datetime_ms_col": "datetime64[ns]",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": False,
        }
        node_func(
            df=sample_df_pd_all_dtypes,
            input_has_schema=_input_has_schema,
            output_has_schema=_output_has_schema,
        )

    def test_relax_false(self, dummy_pd_df):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {"datetime_col": "datetime64[ns]"},
            "allow_subset": True,
        }
        _output_has_schema = {
            "expected_schema": {"datetime_col": "datetime64[ns]"},
            "allow_subset": True,
        }
        with pytest.raises(ValueError, match=r"Diffs displayed only"):
            node_func(
                df=dummy_pd_df,
                input_has_schema=_input_has_schema,
                output_has_schema=_output_has_schema,
            )

    def test_relax_true(self, sample_df_pd_all_dtypes):
        _input_has_schema = {
            "df": "df",
            "expected_schema": {
                "float_col": "float32",
                "int_col": "int16",
                "datetime_col": "datetime64",
                "date_col": "object",
                "string_col": "object",
                "datetime_ms_col": "datetime64[ns]",
            },
            "allow_subset": False,
            "raise_exc": True,
            "relax": True,
        }
        node_func(df=sample_df_pd_all_dtypes, input_has_schema=_input_has_schema)

    def test_full_example_multiple_inputs_outputs(self, sample_df_pd_all_dtypes):
        def _ip_has_schema(df):
            return {
                "df": df,
                "expected_schema": {
                    "int_col": "int64",
                    "string_col": "object",
                    "float_col": "float64",
                    "date_col": "object",
                    "datetime_col": "datetime64[ns]",
                },
                "allow_subset": True,
                "raise_exc": True,
                "relax": False,
            }

        def _op_has_schema(x):
            return {
                "expected_schema": {
                    "int_col": "int64",
                    "string_col": "object",
                    "float_col": "float64",
                    "date_col": "object",
                    "datetime_col": "datetime64[ns]",
                },
                "allow_subset": True,
                "raise_exc": True,
                "relax": False,
                "output": x,
            }

        _input_has_schema = [_ip_has_schema("df1"), _ip_has_schema("df2")]
        _output_has_schema = [_op_has_schema(0), _op_has_schema(1)]
        node_func_full_example_multiple(
            df1=sample_df_pd_all_dtypes,
            df2=sample_df_pd_all_dtypes,
            input_has_schema=_input_has_schema,
            output_has_schema=_output_has_schema,
        )
