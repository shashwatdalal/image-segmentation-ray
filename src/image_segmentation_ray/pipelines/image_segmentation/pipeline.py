"""
This is a boilerplate pipeline 'image_segmentation'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from src.image_segmentation_ray.src.data_bowl.src import (
    simplify_data_structure,
    metadata,
)
from src.image_segmentation_ray.src.image_segmentation.v1.core import (
    train_val_test_split,
)
from src.image_segmentation_ray.src.image_segmentation.v1.nodes.model import (
    trainer,
    inference,
)
from src.image_segmentation_ray.src.image_segmentation.v1.nodes import evaluation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=simplify_data_structure.simplify_data_structure,
                inputs={"input_path": "params:data_bowl.input_path"},
                outputs="data_bowl.simplify_data_structure",
                name="data_bowl.simplify_data_structure",
            ),
            node(
                func=metadata.generate_metadata_table,
                inputs={
                    "input_path": "params:data_bowl.input_path",
                    "simplify_data_structure": "data_bowl.simplify_data_structure",
                },
                outputs="data_bowl.metadata",
                name="data_bowl.metadata",
            ),
            node(
                func=train_val_test_split.train_val_test_column,
                inputs={
                    "df": "data_bowl.metadata",
                    "train_ratio": "params:data_bowl.data_split.train_ratio",
                    "val_ratio": "params:data_bowl.data_split.val_ratio",
                    "split_col_name": "params:data_bowl.data_split.split_col_name",
                    "split_labels": "params:data_bowl.data_split.split_labels",
                },
                outputs="data_bowl.metadata_w_data_split",
                name="data_bowl.metadata_w_data_split",
            ),
            node(
                func=trainer.train,
                inputs={
                    "metadata": "data_bowl.metadata_w_data_split",
                    "split_col_name": "params:data_bowl.data_split.split_col_name",
                    "split_labels": "params:data_bowl.data_split.split_labels",
                    "data_loader_kwargs": "params:data_bowl.train.data_loader_kwargs",
                    "model": "params:data_bowl.model",
                    "loss": "params:data_bowl.loss",
                    "learning_rate": "params:data_bowl.learning_rate",
                    "trainer_kwargs": "params:data_bowl.train.trainer_kwargs",
                    "transforms": "params:data_bowl.train.transforms",
                    "target_transforms": "params:data_bowl.train.target_transforms",
                },
                outputs="data_bowl.trained_model",
                name="data_bowl.trained_model",
            ),
            node(
                func=inference.predict_masks,
                inputs={
                    "metadata": "data_bowl.metadata_w_data_split",
                    "model": "data_bowl.trained_model",
                    "threshold": "params:data_bowl.inference.threshold",
                    "output_layer_activ_function": "params:data_bowl.inference.output_layer_activ_function",
                    "post_process_functions": "params:data_bowl.post_process_functions",
                    "split_col_name": "params:data_bowl.data_split.split_col_name",
                    "splits_to_predict": "params:data_bowl.inference.splits_to_predict",
                    "use_gpu": "params:data_bowl.inference.use_gpu",
                },
                outputs="data_bowl.inference",
                name="data_bowl.inference",
            ),
            node(
                func=evaluation.evaluate_predictions,
                inputs={
                    "evaluation_metrics": "params:data_bowl.evaluate.evaluation_metrics",
                    "metadata": "data_bowl.metadata_w_data_split",
                    "splits_to_predict": "params:data_bowl.inference.splits_to_predict",
                },
                outputs="data_bowl.metadata_w_performance",
                name="data_bowl.metadata_w_performance",
            ),
        ]
    )
