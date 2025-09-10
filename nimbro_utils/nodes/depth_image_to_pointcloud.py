#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import PointCloud2

from nimbro_utils.utility.node import start_and_spin_node
from nimbro_utils.utility.geometry import numpy_to_pointcloud
from nimbro_utils.node_extensions.logger import Logger
from nimbro_utils.node_extensions.depth_converter import DepthConverter
from nimbro_utils.node_extensions.sensor_interface import SensorInterface
from nimbro_utils.node_extensions.parameter_handler import ParameterHandler

### <Parameter Defaults>

node_name = "depth_to_cloud"
severity = 20
throttle = 1.0 # seconds

depth_info_topic = "/gemini/depth/camera_info"
depth_image_topic = ["/gemini/depth/image_raw", "Image", "16UC1"]
publish_interval = 0.1 # seconds

### </Parameter Defaults>

class DepthImageToPointcloud(Node):

    def __init__(self, name=node_name, *, context=None, **kwargs):
        super().__init__(name, context=context, **kwargs)

        self.node_name = self.get_name()
        self.node_namespace = self.get_namespace()

        self._logger = Logger(self)

        # Declare parameters

        self.parameter_handler = ParameterHandler(self)

        self.parameter_handler.declare(
            name="severity",
            dtype=int,
            default_value=severity,
            description="Logging severity of node logger.",
            read_only=False,
            range_min=10,
            range_max=50,
            range_step=10
        )

        self.parameter_handler.declare(
            name="throttle",
            dtype=float,
            default_value=throttle,
            description="Interval in seconds in which logging messages are throttled.",
            read_only=False,
            range_min=0.0,
            range_max=10.0,
            range_step=0.0
        )

        self.parameter_handler.declare(
            name="depth_info_topic",
            dtype=str,
            default_value=depth_info_topic,
            description="Camera info topic corresponding to the depth image topic.",
            read_only=True
        )

        self.parameter_handler.declare(
            name="depth_image_topic",
            dtype=list[str],
            default_value=depth_image_topic,
            description="Depth image topic containing the data to be converted to pointclouds, the corresponding topic type and format retrieved.",
            read_only=False
        )

        self.parameter_handler.declare(
            name="publish_interval",
            dtype=float,
            default_value=publish_interval,
            description="Interval in seconds in which test images are published.",
            read_only=False,
            range_min=0.001,
            range_max=10.0,
            range_step=0.0
        )

        # Variables

        self.stamp_last_publish = None
        self.sensors = SensorInterface(self, settings={
            'names': ["depth"],
            'topics': [self.parameters.depth_image_topic[0]],
            'types': [self.parameters.depth_image_topic[1]],
            'formats': [self.parameters.depth_image_topic[2]]
        })
        self.depth_converter = DepthConverter(self, settings={
            'depth_info_topic': self.parameters.depth_info_topic
        })

        # Create interfaces

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_cloud = self.create_publisher(PointCloud2, f"{self.node_namespace}/{self.node_name}/cloud".replace("//", "/"), qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
        self.timer_image = self.create_timer(0.0, self.publish_image, callback_group=MutuallyExclusiveCallbackGroup())

        self._logger.info("Node started")

    def __del__(self):
        self._logger.info("Node shutdown")

    def filter_parameter(self, name, value, is_declared):
        message = None

        if name == "severity":
            self._logger.set_settings(settings={'severity': value})
        else:
            self.stamp_parameter_reset = self.get_clock().now()
            self.clouds_parameter_reset = 0

        if name == "depth_image_topic":
            if len(value) != 2:
                value = None
                message = f"List must have two elements but it has '{len(value)}'."
            elif hasattr(self, "sensors"):
                self.sensors.set_settings({'topics': [value[0]], 'types': [value[1]], 'formats': [value[2]]})

        return value, message

    def publish_image(self):
        self.depth_converter.wait_for_camera_info()
        if self.stamp_last_publish is None or (self.get_clock().now() - self.stamp_last_publish).nanoseconds * 0.000000001 >= self.parameters.publish_interval:
            success, _, depth_msg = self.sensors.get_data(source="depth")
            if success:
                cloud_np = self.depth_converter.get_point_cloud(pixels=self.depth_converter.get_pixels(), depth_msg=depth_msg, filter_invalid=True)
                cloud_msg = numpy_to_pointcloud(points_3d=cloud_np, header=depth_msg.header)
                self.pub_cloud.publish(cloud_msg)
                self.stamp_last_publish = self.get_clock().now()
                self.clouds_parameter_reset += 1
                if self.clouds_parameter_reset > 1:
                    real_fps = self.clouds_parameter_reset / ((self.stamp_last_publish - self.stamp_parameter_reset).nanoseconds * 0.000000001)
                    real_fps = f"{real_fps:.2f}Hz"
                else:
                    real_fps = "n/a"
                self._logger.info(f"Publishing pointcloud at targetted '{1 / self.parameters.publish_interval:.2f}Hz' (Measured: '{real_fps}')", throttle_duration_sec=self.parameters.throttle)

def main(args=None):
    start_and_spin_node(DepthImageToPointcloud, args=args)

if __name__ == '__main__':
    main()
