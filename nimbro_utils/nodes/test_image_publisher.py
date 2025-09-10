#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image

from nimbro_utils.utility.node import start_and_spin_node
from nimbro_utils.utility.image import convert_image
from nimbro_utils.node_extensions.logger import Logger
from nimbro_utils.node_extensions.parameter_handler import ParameterHandler

### <Parameter Defaults>

node_name = "test_image_publisher"
severity = 20
throttle = 1.0 # seconds

image_width = 1920 # pixels
image_height = 1080 # pixels
image_encoding = "rgb8" # Image encoding supported by convert_image()
publish_interval = 0.1 # seconds

### </Parameter Defaults>

class TestImagePublisher(Node):

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
            name="image_width",
            dtype=int,
            default_value=image_width,
            description="Width of the published test image in pixels.",
            read_only=False,
            range_min=1,
            range_max=10000,
            range_step=1
        )

        self.parameter_handler.declare(
            name="image_height",
            dtype=int,
            default_value=image_height,
            description="Height of the published test image in pixels.",
            read_only=False,
            range_min=1,
            range_max=10000,
            range_step=1
        )

        self.parameter_handler.declare(
            name="image_encoding",
            dtype=str,
            default_value=image_encoding,
            description="Image encoding supported by `nimbro_utils.utility.imagesconvert_image()`.",
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

        # Create interfaces

        qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_image = self.create_publisher(Image, f"{self.node_namespace}/{self.node_name}/image".replace("//", "/"), qos_profile=qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
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
            self.images_parameter_reset = 0

        return value, message

    def publish_image(self):
        if self.stamp_last_publish is None or (self.get_clock().now() - self.stamp_last_publish).nanoseconds * 0.000000001 >= self.parameters.publish_interval:
            image_np = (np.random.rand(self.parameters.image_height, self.parameters.image_width) * 255).astype(np.uint8)
            image_msg = convert_image(image=image_np, target_format="Image", target_encoding=self.parameters.image_encoding, logger=self._logger)
            self.stamp_last_publish = self.get_clock().now()
            image_msg.header.stamp = self.stamp_last_publish.to_msg()
            self.pub_image.publish(image_msg)
            self.images_parameter_reset += 1
            if self.images_parameter_reset > 1:
                real_fps = self.images_parameter_reset / ((self.stamp_last_publish - self.stamp_parameter_reset).nanoseconds * 0.000000001)
                real_fps = f"{real_fps:.2f}Hz"
            else:
                real_fps = "n/a"
            self._logger.info(f"Publishing random images with '{image_np.shape[1]}x{image_np.shape[0]}px' at targeted '{1 / self.parameters.publish_interval:.2f}Hz' (Measured: '{real_fps}')", throttle_duration_sec=self.parameters.throttle)

def main(args=None):
    start_and_spin_node(TestImagePublisher, args=args)

if __name__ == '__main__':
    main()
