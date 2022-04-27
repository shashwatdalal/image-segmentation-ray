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

"""Decorator to remove columns with specific prefix."""
import logging
from typing import Union

import pandas as pd
import pyspark
from boltons.funcutils import wraps

logger = logging.getLogger(__file__)


def remove_input_debug_columns(
    prefix: str = "_",
):  # pylint: disable=missing-return-type-doc
    """Removes all columns with a given prefix from all dataframe inputs.

    Args:
        prefix: The column prefix to remove. Defaults to underscore "_"

    Returns:
        Wrapper function
    """

    def _decorate(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            all_args = []
            for arg in args:
                arg = _remove_debug_columns(arg, prefix)
                all_args.append(arg)

            all_kwargs = {}
            for key, value in kwargs.items():
                value = _remove_debug_columns(value, prefix)
                all_kwargs[key] = value

            result = func(*all_args, **all_kwargs)

            return result

        return _wrapper

    return _decorate


def remove_output_debug_columns(
    prefix: str = "_",
):  # pylint: disable=missing-return-type-doc
    """Removes all columns with a given prefix from all dataframe outputs.

    Args:
        prefix: The column prefix to remove. Defaults to underscore "_"

    Returns:
        Wrapper function
    """

    def _decorate(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):

            result = func(*args, **kwargs)

            if isinstance(result, list):
                result = [_remove_debug_columns(df, prefix) for df in result]
            else:
                result = _remove_debug_columns(result, prefix)

            return result

        return _wrapper

    return _decorate


def _remove_debug_columns(
    df: Union[pyspark.sql.DataFrame, pd.DataFrame], prefix: str
) -> Union[pyspark.sql.DataFrame, pd.DataFrame]:
    """Removes columns that starts with a given prefix."""
    if isinstance(df, pyspark.sql.DataFrame):
        non_debug_cols = [x for x in df.columns if not x.startswith(prefix)]
        df = df.select(*non_debug_cols)

        # pylint: disable=logging-fstring-interpolation
        logger.info(f"Removing debug columns from spark dataframe: {df}")
    elif isinstance(df, pd.DataFrame):
        non_debug_cols = [x for x in df.columns if not x.startswith(prefix)]
        df = df[non_debug_cols]

        # pylint: disable=logging-fstring-interpolation
        logger.info(f"Removing debug columns from pandas dataframe: {df}")
    else:
        logger.info("Non spark or pandas dataframe: %s", df)

    return df
