from glob import glob

from setuptools import setup

package_name = "uav_flight_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="andy",
    maintainer_email="music-wang@users.noreply.github.com",
    description="PX4 uXRCE-DDS 飞控接口：视觉引导速度控制 + 安全起飞序列",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "flight_control = uav_flight_control.flight_control_node:main",
        ],
    },
)
