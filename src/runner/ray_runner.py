import warnings
from typing import Dict, Any

from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner
import ray
from pluggy import PluginManager


class RayRunner(SequentialRunner):
    def __init__(self, ray_init_args):
        """Instantiates the executor class.
        Args:
            is_async:
            nodes: The iterable of nodes to run.
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
        """
        super().__init__()
        self._ray_config: Dict[str, Any] = {}
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ray.init(**ray_init_args)

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str = None,
    ) -> None:
        """
        This doesn't do _any_ hooks at the moment

        Args:
            **kwargs:
        """
        # check that the node function is a ray function
        nodes = pipeline.nodes
        node_funcs = [node.func for node in nodes]
        decorated_node_funcs = []
        for _f in node_funcs:
            if type(_f) == ray.remote_function.RemoteFunction:
                decorated_node_funcs.append(_f)
            else:
                decorated_node_funcs.append(ray.remote(num_gpus=4)(_f))

        for _df, _node in zip(decorated_node_funcs, nodes):
            # run serially
            self._logger.info("Ray Executor reading %s input", _node.name)
            input_pointers = [remote_load.remote(catalog, _input) for _input in _node.inputs]
            self._logger.info("Submitting node %s to Ray Executor", _node.name)
            # assume only single output
            result = _df.remote(*input_pointers)
            self._logger.info("Ray Executor writing %s output", _node.name)
            # block
            ray.get(remote_save.remote(catalog, _node.outputs[0], result))

@ray.remote
def remote_load(catalog, input):
    return catalog.load(input)

@ray.remote
def remote_save(catalog, name, result):
    catalog.save(name, result)