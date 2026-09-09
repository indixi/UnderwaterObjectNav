#!/usr/bin/env python3
"""Replace this system-Python process with the configured Conda worker."""

import argparse
import os
import sys


def non_ros_arguments(arguments):
    """Remove roslaunch remappings such as ``__name:=`` and ``__log:=``."""
    return [argument for argument in arguments if ":=" not in argument]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-executable", required=True)
    parser.add_argument("--conda-env", required=True)
    parser.add_argument("--bc-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29500)
    args = parser.parse_args(non_ros_arguments(sys.argv[1:]))

    command = [
        args.conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "-u",
        "-m",
        "objectnav_bc.online.inference_worker",
        "--bc-config",
        args.bc_config,
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.device,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    environment = os.environ.copy()
    # The worker deliberately does not import ROS.  Replacing PYTHONPATH keeps
    # Noetic's Python-3.8 packages out of the Python-3.10 Conda interpreter,
    # while still making the source-tree objectnav_bc package importable.
    environment["PYTHONPATH"] = os.path.abspath(args.package_root)
    os.execve(args.conda_executable, command, environment)


if __name__ == "__main__":
    main()
