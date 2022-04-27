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
# pylint: disable=missing-return-doc, missing-return-type-doc, line-too-long

"""Decorator to check primary key of function output."""
import functools
import logging
from typing import List, Union

import pandas as pd
import pyspark

from ..internals import _get_param_from_arg_and_kwarg

logger = logging.getLogger(__name__)

OUTPUT_PRIMARY_KEY = "output_primary_key"


def _add_output_primary_key(func, *args, **kwargs):
    """Filter args or kwargs.

    Raises:
        KeyError: If kwarg_filter key does not match any kwarg key.
    """
    output_primary_key, args, kwargs = _get_param_from_arg_and_kwarg(
        OUTPUT_PRIMARY_KEY, *args, **kwargs
    )

    if output_primary_key:
        columns_list = output_primary_key["columns"]
        nullable = output_primary_key.get("nullable", False)

        def wrapper(*args, **kwargs):

            result_df = func(*args, **kwargs)

            logger.info(
                "Checking primary key validation on output dataframe - "
                "Non Duplicate Check & Not Null Check",
            )

            _duplicate_and_null_check(result_df, columns_list, nullable)

            return result_df

        return wrapper, args, kwargs
    return func, args, kwargs


def add_output_primary_key():
    """Primary key check function decorator.

    Returns:
        Wrapper function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            new_func, args, kwargs = _add_output_primary_key(func, *args, **kwargs)
            result_df = new_func(*args, **kwargs)

            return result_df

        return wrapper

    return decorate


def _duplicate_and_null_check(
    result_df: Union[pyspark.sql.DataFrame, pd.DataFrame],
    columns_list: List[str],
    nullable: bool,
    df_name: str = "0",
):
    """Duplicate and nullable check on primary key columns of a dataframe."""
    if isinstance(result_df, pyspark.sql.DataFrame):
        actual_count = result_df.count()
        if not nullable:
            # Not allowing any duplicates and any nulls in primary key
            expected_count = (
                result_df.select(*columns_list)
                .dropDuplicates()
                .dropna(how="any")
                .count()
            )

        else:
            # Not allowing any duplicates and allowing a null in a composite key
            expected_count = (
                result_df.select(*columns_list)
                .dropDuplicates()
                .dropna(how="all")
                .count()
            )

    elif isinstance(result_df, pd.DataFrame):
        actual_count = result_df.shape[0]
        if not nullable:
            # Not allowing any duplicates and any nulls in primary key
            expected_count = (
                result_df[columns_list].drop_duplicates().dropna(how="any").shape[0]
            )

        else:
            # Not allowing any duplicates and allowing a null in a composite key
            expected_count = (
                result_df[columns_list].drop_duplicates().dropna(how="all").shape[0]
            )

    else:
        raise ValueError("`result_df` should be of type pandas or spark dataframe.")

    pk_check_failed = bool(actual_count != expected_count)

    logger.info(
        "Running primary key check.",
        extra={
            "primary_key": {  # contain everything under a single payload
                "actual_count": actual_count,
                "expected_count": expected_count,
                "pk_check_failed": str(pk_check_failed),
                "df_name": df_name,
            }
        },
    )

    if pk_check_failed:
        raise TypeError(
            f"Primary key columns {columns_list} has either duplicate values "
            f"or null values."
        )
