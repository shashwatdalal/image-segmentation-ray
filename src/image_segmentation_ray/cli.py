from kedro.framework.cli.utils import _get_values_as_tuple
from kedro.framework.session import KedroSession
from kedro.utils import load_obj

"""Command line tools for manipulating a Kedro project.
Intended to be invoked via `kedro`."""
import click
from kedro.framework.cli.utils import (
    CONTEXT_SETTINGS,
    _config_file_callback,
    _reformat_load_versions,
    _split_params,
    env_option,
    split_string,
)

from kedro.framework.cli.project import (
    FROM_INPUTS_HELP,
    TO_OUTPUTS_HELP,
    FROM_NODES_HELP,
    TO_NODES_HELP,
    NODE_ARG_HELP,
    RUNNER_ARG_HELP,
    ASYNC_ARG_HELP,
    TAG_ARG_HELP,
    LOAD_VERSION_HELP,
    PIPELINE_ARG_HELP,
    CONFIG_FILE_HELP,
    PARAMS_ARG_HELP,
)


@click.group(context_settings=CONTEXT_SETTINGS, name=__file__)
def cli():
    """Command line tools for manipulating a Kedro project."""


@cli.command()
@click.option(
    "--from-inputs", type=str, default="", help=FROM_INPUTS_HELP, callback=split_string
)
@click.option(
    "--to-outputs", type=str, default="", help=TO_OUTPUTS_HELP, callback=split_string
)
@click.option(
    "--from-nodes", type=str, default="", help=FROM_NODES_HELP, callback=split_string
)
@click.option(
    "--to-nodes", type=str, default="", help=TO_NODES_HELP, callback=split_string
)
@click.option("--node", "-n", "node_names", type=str, multiple=True, help=NODE_ARG_HELP)
@click.option(
    "--runner", "-r", type=str, default=None, multiple=False, help=RUNNER_ARG_HELP
)
@click.option("--async", "is_async", is_flag=True, multiple=False, help=ASYNC_ARG_HELP)
@env_option
@click.option("--tag", "-t", type=str, multiple=True, help=TAG_ARG_HELP)
@click.option(
    "--load-version",
    "-lv",
    type=str,
    multiple=True,
    help=LOAD_VERSION_HELP,
    callback=_reformat_load_versions,
)
@click.option("--pipeline", "-p", type=str, default=None, help=PIPELINE_ARG_HELP)
@click.option(
    "--params", type=str, default="", help=PARAMS_ARG_HELP, callback=_split_params
)
def run(
    tag,
    env,
    runner,
    is_async,
    node_names,
    to_nodes,
    from_nodes,
    from_inputs,
    to_outputs,
    load_version,
    pipeline,
    params,
):
    """Run the pipeline."""
    runner = runner or "SequentialRunner"

    tag = _get_values_as_tuple(tag) if tag else tag
    node_names = _get_values_as_tuple(node_names) if node_names else node_names

    with KedroSession.create(env=env, extra_params=params) as session:
        context = session.load_context()
        runner_instance = _instantiate_runner(runner, context)
        session.run(
            pipeline_name=pipeline,
            tags=tag,
            runner=runner_instance,
            node_names=node_names,
            from_nodes=from_nodes,
            to_nodes=to_nodes,
            from_inputs=from_inputs,
            to_outputs=to_outputs,
            load_versions=load_version,
        )


def _instantiate_runner(runner, project_context):
    runner_class = load_obj(runner, "kedro.runner")

    if runner.endswith("RayRunner"):
        ray_init_args = project_context.params.get("ray_init") or {}
        return runner_class(ray_init_args)
    else:
        return runner_class()
