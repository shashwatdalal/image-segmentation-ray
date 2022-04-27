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

"""Data Science BOWL connector."""

import os

import pandas as pd


def generate_metadata_table(
    input_path: str, simplify_data_structure: bool
) -> pd.DataFrame:
    """Generate metadata table of "Data Science BOWL" dataset.

    It will contain: image id, image path, mask path and number of nuclei.

    Args:
        input_path: path pointing to dataset location
        simplify_data_structure: dummy parameter to establish dependency
        with upstream node
    Returns:
        Metadata table.
    """
    sub_folders_list = [
        sub_folder for sub_folder in os.listdir(input_path) if sub_folder != ".DS_Store"
    ]

    list_of_entries = []
    for current_item in sub_folders_list:
        list_of_entries.append(
            {
                "id": current_item,
                "img_path": os.path.join(input_path, current_item, "image.png"),
                "mask_path": os.path.join(input_path, current_item, "mask.png"),
                "nb_of_nuclei": len(
                    os.listdir(os.path.join(input_path, current_item, "masks"))
                ),
            }
        )

    data_table = pd.DataFrame(list_of_entries)

    return data_table
