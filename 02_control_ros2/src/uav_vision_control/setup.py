from setuptools import find_packages, setup

package_name = 'uav_vision_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
                'launch/vision_detector.launch.py',
                'launch/mission.launch.py',
            ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andy',
    maintainer_email='music-wang@users.noreply.github.com',
    description='USB 摄像头 + YOLO11(ONNX) 目标检测节点',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_detector = uav_vision_control.vision_detector:main',
            'cell_mapper_node = uav_vision_control.cell_mapper_node:main',
            'serial_bridge = uav_vision_control.serial_bridge_node:main',
        ],
    },
)
