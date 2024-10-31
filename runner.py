"""
Simplified version of cleanRL's benchmarking script.
Iterates over combinations and calls the training with asynchronous workers.
The original version can be found here: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/benchmark.py
"""

import importlib
import random
import shlex
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from itertools import product
from pathlib import Path
from typing import Any

import tyro
from tqdm import tqdm

import wandb
from config.runner_config import Configs, RunnerArgs


def _sanity_checks(config: RunnerArgs, key: str) -> None:
    """Throw some warnings if the settings produce unnecessary runs."""

    if not config.brb.reset_weights:
        if key in {
            "brb.reset_interpolation_factor",
            "brb.reset_embedding",
            "brb.reset_projector",
            "brb.reset_convlayers",
            "brb.reset_batchnorm",
        }:
            warnings.warn(f"UNECESSERY RUNS: reset_weights=False, but runner iterates over settings for {key}.", UserWarning)

    if not config.brb.recluster:
        if key == "brb.reclustering_method" or key == "brb.subsample_size":
            warnings.warn(f"UNECESSERY RUNS: recluster=False, but runner iterates over settings for {key}.", UserWarning)
    else:
        if key == "brb.recalculate_centers":
            warnings.warn("UNECESSERY RUNS: recluster=True. Recalculating centroids has no effect.", UserWarning)

    if config.brb.reset_momentum:
        if "key" == "dc_optimizer.optimizer":
            warnings.warn("UNECESSERY RUNS: reset_momentum=True is only working for momentum_based optimizers.", UserWarning)


def get_settings_combinations(
    config: RunnerArgs,
) -> tuple[list[tuple[tuple[str, Any], ...]], list[tuple[str, Any]]]:
    """Return a list of different combinations of settings to run.
    Each element in the list has the form:
    ( ("setting1", value1), ("setting2", value2), ... )
    """
    # Generate a list of tuples with the
    constant_settings = []
    reset_iterations = []

    def _check_and_append(
        cli_flag: str, field_value: Any, constant_settings: list[tuple[str, Any]], reset_iterations: list[tuple[str, Any]]
    ) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
        """Helper function to check and append settings to the correct list."""
        if isinstance(field_value, tuple):
            # Throw some warnings if the settings produce unnecessary runs
            _sanity_checks(config, cli_flag)
            values = [(cli_flag, item) for item in field_value]
            reset_iterations.append(values)
        # Settings that are the same for all runs and are not yet appended
        elif cli_flag not in constant_settings:
            constant_settings.append((cli_flag, field_value))

        return constant_settings, reset_iterations

    for field in fields(config):
        key = field.name
        field_value = getattr(config, key)

        # Settings that are not being iterated over
        if key == "experiment":
            # Generate a list of tuples with all attributes and their values from the ExperimentArgs config
            # and append them to the constant settings (they're never iterated over)
            constant_settings.extend(
                [
                    (f"{key}.{inner_key}", inner_value)
                    for inner_key, inner_value in vars(field_value).items()
                    if inner_key != "gpu"
                ]
            )

        elif key == "brb" or key == "pretrain_optimizer" or key == "dc_optimizer":
            # Hierarchical settings that are being iterated over
            # Can be done with a single for-loop because we only have one level of hierarchy
            for inner_field in fields(field_value):
                inner_key = inner_field.name
                inner_field_value = getattr(field_value, inner_key)

                # The flag that will actually be used in the command line
                cli_flag = f"{key}.{inner_key}"

                # Check and append the settings to the correct list
                constant_settings, reset_iterations = _check_and_append(
                    cli_flag=cli_flag,
                    field_value=inner_field_value,
                    constant_settings=constant_settings,
                    reset_iterations=reset_iterations,
                )
        elif key == "command" or key == "workers":
            # Runner meta settings that are not used to call train.py
            pass
        else:
            # Single settings that may be iterated over
            constant_settings, reset_iterations = _check_and_append(
                cli_flag=key, field_value=field_value, constant_settings=constant_settings, reset_iterations=reset_iterations
            )

    iteration_settings = list(product(*reset_iterations))

    return iteration_settings, constant_settings


def generate_commands(
    exp_args: RunnerArgs, combinations: list[tuple[tuple[str, Any], ...]], constant_settings: list[tuple[str, Any]]
) -> list[str]:
    """Generates a list of commands to run from a list of settings combinations and a list of settings that are the same for all runs."""

    # List of arguments that are the same for all runs
    fixed_args = [f"--{setting[0]} {setting[1]}" for setting in constant_settings]
    # Handle passing of multiple gpus
    if isinstance(exp_args.experiment.gpu, tuple):
        fixed_args += [f"--experiment.gpu {' '.join([str(gpu) for gpu in exp_args.experiment.gpu])}"]
    else:
        fixed_args += [f"--experiment.gpu {exp_args.experiment.gpu}"]

    commands = []

    api = wandb.Api(timeout=60, overrides={"entity": args.experiment.wandb_entity})
    for settings in tqdm(combinations):
        command_list = [exp_args.command]
        run_config = {}
        none_config = {}
        for setting in settings:
            command_list += [f"--{setting[0]} {setting[1]!r}"]
            if setting[1] is not None:
                run_config[f"config.{setting[0]}"] = setting[1]
            else:
                none_config[f"{setting[0]}"] = setting[1]
        run_config["tags"] = args.experiment.tag
        # Check for whether a run has already been completed and can be skipped
        if args.experiment.wandb_check_duplicates:
            runs = api.runs(args.experiment.wandb_project.replace('"', ""), filters=run_config)
            filtered_runs = []
            for run in runs:
                add = True
                for k in none_config.keys():
                    if run.config[k] is not None:
                        add = False
                if run.state == "crashed" or run.state == "killed":
                    add = False
                if add:
                    filtered_runs.append(run)

            if len(filtered_runs) > 0:
                continue

        full_command = command_list + fixed_args
        commands += [" ".join(full_command)]

    print(f"TOTAL remaining runs: {len(commands)} of {len(combinations)}")
    return commands


def run_experiment(command: str) -> bool:
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


def load_configs_from_files(path: str) -> dict:
    config_data = {}

    # List all Python files in the specified directory
    for rel_path in Path(path).rglob("*.py"):
        # Create a full file path
        filepath = rel_path.resolve()
        # Create a module name based on the file name
        module_name = filepath.stem

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Checking for a config attribute in the module allows having other config files in the same directory
        if hasattr(module, "config"):
            config_data |= module.config

    return config_data


if __name__ == "__main__":
    Configs |= load_configs_from_files("config")
    Configs = tyro.extras.subcommand_type_from_defaults(Configs)

    args = tyro.cli(Configs)

    combinations, constant_settings = get_settings_combinations(args)

    commands = generate_commands(args, combinations, constant_settings)

    settings_to_iterate = set([inner[0] for tup in combinations for inner in tup])
    print("#" * 70)
    print(f"ITERATING OVER COMBINATIONS OF: {settings_to_iterate}. TOTAL COMBINATIONS: {len(commands)}.")
    print("#" * 70)

    if args.workers > 0:
        executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="brb-thread")
        for command in commands:
            # Randomize starting times to not initially schedule everything on the GPU with the lowest ID
            time.sleep(random.randint(5, 10))
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print("not running the experiments because --workers is set to 0; just printing the commands to run")
