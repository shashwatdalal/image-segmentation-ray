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
"""Dataset."""

import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    """This class implements a PyTorch Dataset.

    Images and masks are loaded based on their path in the metadata
    csv. The transforms defined here are applied on the fly on the
    batch loaded by DataLoader (not all the data at once).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        transforms: torchvision.transforms.Compose = None,
        target_transforms: torchvision.transforms.Compose = None,
    ):
        """Ran once when instantiating the Dataset object.

        It initializes :
        - table containing images and masks paths.
        - transforms to be applied on images
        - target transforms to be applied on masks.
        """
        super().__init__()
        self.metadata = metadata
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        """Returns the number of samples in our dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int):
        """Loads and returns a sample from the dataset at the given index idx.

        Based on the index, it identifies the image’s and mask's location
        on disk, converts them to tensors using read_image, call the transforms
        functions on them, and returns the updated tensors in a tuple.

        Args:
            idx: image and mask index
        Returns:
            image: transformed image
            mask: transformed mask.
        """
        img_path = self.metadata.iloc[idx].img_path
        image = read_image(img_path)

        # The output of read_image is in uint8 type
        # pylint: disable=no-member
        image = image.type(torch.float)

        # Binarize mask
        mask = read_image(self.metadata.iloc[idx].mask_path) > 0

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            mask = self.target_transforms(mask)

        return image, mask
