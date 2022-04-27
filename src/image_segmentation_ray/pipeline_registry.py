"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from src.image_segmentation_ray.pipelines.image_segmentation import create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": create_pipeline()}
