import os
from glob import glob

from setuptools import find_packages
from setuptools import setup

package_name = "helhest_stack_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ales Kucera",
    maintainer_email="kuceral4@fel.cvut.cz",
    description="Odin's on-robot mapper and planner (elevation_node), ROS 2.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "elevation_node = helhest_stack_ros.elevation_node:main",
        ],
    },
)
