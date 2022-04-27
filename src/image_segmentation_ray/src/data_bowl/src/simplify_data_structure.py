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
from multiprocessing import Pool

import numpy as np
from skimage import io
from tqdm.contrib.concurrent import process_map
from functools import partial
import warnings


from src.image_segmentation_ray.src.image_segmentation.v1.core.preprocessing.formatting import (
    combine_masks_into_single_mask,
)


def simplify_data_structure(input_path: str, pool_size=2) -> bool:
    """The Data Science Bowl dataset comprises many folders with different ID's.

    In all these folders, the structure is the same :
    some_ID
       |
       |---images
       |      |---some_ID.png
       |---masks
              |---mask_1_ID.png
              |---mask_2_ID.png
              |---mask_3_ID.png

    This function will simplify the structure by :
    - combining masks together into a single one
    - applying a less nested structure
    - adding much shorter ID's to simplify paths

    New structure of each folder:
    id_1
     |
     |---image.png
     |---mask.png

    As Kedro doesn't provide the functionality to output many images at once,
    we'll have writing operations inside the node.

    Args:
        input_path: path pointing to dataset location
    Returns:
        A dummy boolean to establish dependency with downstream node
    """
    # Generate the list of sub-folders (ignoring .DS_Store folder)
    sub_folders_list = [
        sub_folder for sub_folder in os.listdir(input_path) if sub_folder != ".DS_Store"
    ]

    process_map(partial(simplify_image, input_path), sub_folders_list, max_workers=pool_size)
    return True


def simplify_image(input_path, id):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        masks_list = []

        # For each sub-folder "id", get the list of masks inside "masks" folder
        sub_folder_path = os.path.join(input_path, id)
        for mask in os.listdir(os.path.join(sub_folder_path, "masks")):
            if mask.lower().endswith(".png"):
                masks_list.append(
                    io.imread(os.path.join(sub_folder_path, "masks", mask))
                )

        # Generate output mask path
        output_mask_path = os.path.join(sub_folder_path, "mask.png")

        # Combine the masks list into a single mask
        output_mask = combine_masks_into_single_mask(masks_list)

        # Write output mask
        io.imsave(output_mask_path, output_mask.astype(np.uint8))

        # All channels in the image are carrying same information
        image = io.imread(os.path.join(sub_folder_path, "images", id + ".png"))
        first_channel = image[:, :, 0]
        image = np.reshape(
            first_channel, newshape=(first_channel.shape[0], first_channel.shape[1], 1)
        )
        io.imsave(os.path.join(sub_folder_path, "image.png"), image.astype(np.uint8))