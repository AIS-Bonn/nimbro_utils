from setuptools import setup, find_packages

package_name = "nimbro_utils"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(include=[f"{package_name}*"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bastian Pätzold",
    maintainer_email="paetzold@ais.uni-bonn.de",
    description="A diverse collection of ROS2 and robotics-related utilities.",
    license_files=["LICENSE"],
    entry_points={
        "console_scripts": [
            f"test_image_publisher = {package_name}.nodes.test_image_publisher:main",
            f"depth_image_to_pointcloud = {package_name}.nodes.depth_image_to_pointcloud:main"
        ]
    }
)
