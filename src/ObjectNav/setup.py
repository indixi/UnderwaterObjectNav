#!/usr/bin/env python3

from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=[
        "objectnav_bc",
        "objectnav_bc.dataset",
        "objectnav_bc.eval",
        "objectnav_bc.mapping",
        "objectnav_bc.models",
        "objectnav_bc.online",
        "objectnav_bc.perception",
        "objectnav_bc.train",
    ],
    package_dir={"": "."},
)

setup(**setup_args)
