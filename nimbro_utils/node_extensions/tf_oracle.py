#!/usr/bin/env python3

import copy

import numpy as np

import rclpy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, Point, PointStamped, Pose, PoseStamped

from nimbro_utils.node_extensions.logger import Logger
from nimbro_utils.utility.misc import assert_type_value, assert_keys, update_dict
from nimbro_utils.utility.geometry import do_transform, convert_transform, create_point

default_settings = {
    # Logger severity in [10, 20, 30, 40, 50] (int) or attribute name (str) of another Logger relative to parent node (supports nested attributes separated by dots).
    'severity': 20,
    # Logger suffix (str).
    'suffix': "tf_oracle",
    # Name of the 'tf2_ros.buffer.Buffer' attribute of the parent node (str). If not available, it is added automatically.
    'tf_buffer': "tf_buffer",
    # Name of the 'tf2_ros.transform_listener.TransformListener' attribute of the parent node (str). If not available, it is added automatically.
    'tf_listener': "tf_listener",
}

class TfOracle:
    """
    A utility class for querying and transforming geometry using tf2 tree.

    Provides helper methods to:
      - Query transforms between frames with timeout and freshness control.
      - Transform geometry messages (points, poses, clouds) to different frames.
      - Interpolate or measure distances between points, including TF frame resolution.

    Automatically initializes tf2 buffer and listener if not already present on the node.
    """

    def __init__(self, node, settings=None):
        """
        Initialize the TfOracle and its associated TF2 utilities.

        Args:
            node (rclpy.node.Node): The ROS2 node to attach to.
            settings (dict | None, optional): Configuration settings to override defaults. Defaults to None.

        Raises:
            AssertionError: If input arguments are invalid.
        """
        # node
        assert_type_value(
            obj=node,
            type_or_value=rclpy.node.Node,
            name="argument 'node'"
        )
        self._node = node

        # logger:
        self._logger = Logger(self._node, settings={
            'severity': default_settings['severity'],
            'prefix': None,
            'name': default_settings['suffix']
        })

        # settings
        settings = update_dict(old_dict=default_settings, new_dict=settings)
        self.set_settings(settings=settings, keep_existing=False)

    # settings

    def get_settings(self):
        """
        Retrieve the current settings of the SensorInterface.

        Returns:
            dict: A deep copy of the current settings.
        """
        return copy.deepcopy(self._settings)

    def set_settings(self, settings, keep_existing=True):
        """
        Update settings of the SensorInterface.

        Args:
            settings (dict): New settings to apply.
            keep_existing (bool, optional): If True, merge with existing settings. Otherwise, replace current settings entirely. Defaults to True.

        Raises:
            AssertionError: If input arguments or provided settings are invalid.
        """
        # parse arguments
        assert_type_value(obj=keep_existing, type_or_value=bool, name="argument 'keep_existing'", logger=self._logger)
        settings = update_dict(old_dict=self._settings if keep_existing else {}, new_dict=settings, key_name="setting", logger=self._logger, info=False, debug=False)
        assert_keys(obj=settings, keys=default_settings.keys(), mode="match", name="settings", logger=self._logger)

        # Logger
        self._logger.set_settings({'severity': settings['severity'], 'name': settings['suffix']})

        # tf2_ros.buffer.Buffer
        assert_type_value(obj=settings['tf_buffer'], type_or_value=str, name="setting 'tf_buffer'", logger=self._logger)
        if hasattr(self, "_settings"):
            if settings['tf_buffer'] != self._settings['tf_buffer']:
                assert_type_value(obj=self._node.__getattribute__(self._settings['tf_buffer']), type_or_value=tf2_ros.buffer.Buffer, name=f"parent node attribute '{self._settings['tf_buffer']}' (pointed to by old setting 'tf_buffer')", logger=self._logger)
                delattr(self._node, self._settings['tf_buffer'])
        if hasattr(self._node, settings['tf_buffer']):
            assert_type_value(obj=self._node.__getattribute__(settings['tf_buffer']), type_or_value=tf2_ros.buffer.Buffer, name=f"parent node attribute '{settings['tf_buffer']}' (pointed to by setting 'tf_buffer')", logger=self._logger)
        else:
            self._logger.debug(f"Adding 'tf2_ros.buffer.Buffer' attribute '{settings['tf_buffer']}' to parent node")
            setattr(self._node, settings['tf_buffer'], tf2_ros.buffer.Buffer()) # cache_time=rclpy.duration.Duration(seconds=100))
        self._tf_buffer = self._node.__getattribute__(settings['tf_buffer'])

        # tf2_ros.transform_listener.TransformListener
        assert_type_value(obj=settings['tf_listener'], type_or_value=str, name="setting 'tf_listener'", logger=self._logger)
        if hasattr(self, "_settings"):
            if settings['tf_listener'] != self._settings['tf_listener']:
                assert_type_value(obj=self._node.__getattribute__(self._settings['tf_listener']), type_or_value=tf2_ros.transform_listener.TransformListener, name=f"parent node attribute '{self._settings['tf_listener']}' (pointed to by old setting 'tf_listener')", logger=self._logger)
                delattr(self._node, self._settings['tf_listener'])
        if hasattr(self._node, settings['tf_listener']):
            assert_type_value(obj=self._node.__getattribute__(settings['tf_listener']), type_or_value=tf2_ros.transform_listener.TransformListener, name=f"parent node attribute '{settings['tf_listener']}' (pointed to by setting 'tf_listener')", logger=self._logger)
        else:
            self._logger.debug(f"Adding 'tf2_ros.transform_listener.TransformListener' attribute '{settings['tf_listener']}' to parent node")
            setattr(self._node, settings['tf_listener'], tf2_ros.transform_listener.TransformListener(self._node.__getattribute__(settings['tf_buffer']), self._node, spin_thread=False))
        self._tf_listener = self._node.__getattribute__(settings['tf_listener'])

        self._settings = settings

    # transform

    def get_transform(self, source_frame, target_frame, *, target_format="TransformStamped", mute_lookup_error=False, time=None, max_age=None, timeout=1.0):
        """
        Lookup the transform between two tf frames, optionally converting format and validating freshness.

        Args:
            source_frame (str): Name of the source frame.
            target_frame (str): Name of the target frame.
            target_format (str, optional): Output format for the transform. Must be supported by `convert_transform()`. Defaults to "TransformStamped".
            mute_lookup_error (bool, optional): If True, suppress log output on lookup errors. Defaults to False.
            time (rclpy.time.Time | None, optional): Specific timestamp to query the transform at. Use None for latest. Defaults to None.
            max_age (rclpy.duration.Duration | float | int | None, optional): Maximum transform age relative to `time`. Use None to skip age check. Defaults to None.
            timeout (rclpy.duration.Duration | float | int | None, optional): Timeout for lookup. Use None to not wait at all. Defaults to 1.0.

        Returns:
            tuple[bool, str, Any]:
                - Success flag.
                - Human-readable message.
                - Transform in the requested format if successful.
        """
        # parse arguments
        assert_type_value(obj=source_frame, type_or_value=str, name="argument 'source_frame'", logger=self._logger)
        assert_type_value(obj=target_frame, type_or_value=str, name="argument 'target_frame'", logger=self._logger)
        assert_type_value(obj=target_format, type_or_value=str, name="argument 'target_format'", logger=self._logger)
        assert_type_value(obj=mute_lookup_error, type_or_value=bool, name="argument 'mute_lookup_error'", logger=self._logger)
        assert_type_value(obj=time, type_or_value=[rclpy.time.Time, None], name="argument 'time'", logger=self._logger)
        assert_type_value(obj=max_age, type_or_value=[rclpy.duration.Duration, float, int, None], name="argument 'max_age'", logger=self._logger)
        assert_type_value(obj=timeout, type_or_value=[rclpy.duration.Duration, float, int, None], name="argument 'timeout'", logger=self._logger)

        # lookup transform

        stamp_before_lookup = self._node.get_clock().now()

        if source_frame == target_frame:
            transform_stamped = TransformStamped()
            transform_stamped.header.frame_id = target_frame
            if time is None:
                transform_stamped.header.stamp = stamp_before_lookup.to_msg()
            else:
                transform_stamped.header.stamp = time.to_msg()
            transform_stamped.child_frame_id = source_frame
            time_waited = 0
            time_age = (stamp_before_lookup - rclpy.time.Time.from_msg(transform_stamped.header.stamp)).nanoseconds / 1e9

        else:
            if time is None:
                time = rclpy.time.Time()
            if max_age is not None:
                if isinstance(max_age, float) or isinstance(max_age, int):
                    max_age = rclpy.duration.Duration(seconds=max_age)
            if timeout is None:
                timeout = rclpy.duration.Duration() # TODO map None to never timeout?
            elif isinstance(timeout, float) or isinstance(timeout, int):
                timeout = rclpy.duration.Duration(seconds=timeout)

            while True:
                try:
                    transform_stamped = self._tf_buffer.lookup_transform(target_frame, source_frame, time=time, timeout=timeout)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    e_fmt = repr(e).replace('"', "'")
                    message = f"Failed to lookup transform from '{source_frame}' to '{target_frame}': {e_fmt}"
                    if not mute_lookup_error:
                        self._logger.error(message)
                    return False, message, None
                else:
                    stamp_after_lookup = self._node.get_clock().now()
                    time_waited = stamp_after_lookup - stamp_before_lookup
                    time_age = stamp_after_lookup - rclpy.time.Time.from_msg(transform_stamped.header.stamp)

                    if time_waited > timeout:
                        message = f"Failed to lookup transform from '{source_frame}' to '{target_frame}' (Timeout after {time_waited.nanoseconds / 1e9})."
                        if not mute_lookup_error:
                            self._logger.error(message)
                        return False, message, None

                    if max_age is not None:
                        if time_age > max_age:
                            if not mute_lookup_error:
                                message = f"Retreived transform from '{source_frame}' to '{target_frame}' that is '{time_age.nanoseconds / 1e9:.3f}s' old (> '{max_age.nanoseconds / 1e9:.3f}s'), waiting for newer transform."
                                self._logger.debug(message, throttle_duration_sec=1.0)
                        else:
                            break
                    else:
                        break

            time_waited = time_waited.nanoseconds / 1e9
            time_age = time_age.nanoseconds / 1e9

        # convert transform
        try:
            transform_stamped = convert_transform(transform_stamped, target_format)
        except Exception as e:
            e_fmt = repr(e).replace('"', "'")
            message = f"Failed in convert_transform(): {e_fmt}"
            self._logger.error(message)
            return False, message, None

        # log
        message = f"Looked up '{time_age:.3f}s' old transform from '{source_frame}' to '{target_frame}' in format '{target_format}' after waiting '{time_waited:.3f}s'."
        self._logger.debug(message)

        return True, message, transform_stamped

    def transform_to_frame(self, point_or_pose_or_cloud, target_frame):
        """
        Transform a geometry object to the target frame using the ft2 tree.

        Args:
            point_or_pose_or_cloud (PointStamped | PoseStamped | PointCloud2): Input geometry in a known frame.
            target_frame (str): Frame to transform the input into.
            # TODO add parameters passed to get_transform: mute_lookup_error, time, max_age, timeout

        Returns:
            tuple[bool, str, Any]:
                - Success flag.
                - Human-readable message.
                - Transformed object if successful.
        """
        # parse arguments
        assert_type_value(obj=point_or_pose_or_cloud, type_or_value=[PointStamped, PoseStamped, PointCloud2], name="argument 'point_or_pose_or_cloud'", logger=self._logger)
        assert_type_value(obj=target_frame, type_or_value=str, name="argument 'target_frame'", logger=self._logger)
        if point_or_pose_or_cloud.header.frame_id == target_frame:
            message = "The frame of 'point_or_pose_or_cloud' equals the target frame."
            self._logger.debug(message)
            return True, message, point_or_pose_or_cloud
        source_frame = point_or_pose_or_cloud.header.frame_id

        # retrieve transform
        success, message, transform = self.get_transform(point_or_pose_or_cloud.header.frame_id, target_frame)
        if not success:
            message = f"Failed in get_transform(): {message}"
            return False, message, None

        # apply transform
        try:
            point_or_pose_or_cloud = do_transform(point_or_pose_or_cloud, transform)
        except Exception as e:
            e_fmt = repr(e).replace('"', "'")
            message = f"Failed in do_transform(): {e_fmt}"
            self._logger.error(message)
            return False, message, None

        # log
        message = f"Transformed provided geometry message from frame '{source_frame}' to frame '{target_frame}'."
        self._logger.debug(message)

        return True, message, point_or_pose_or_cloud

    def get_interpolation(self, point_or_pose_a, point_or_pose_b, *, mode='mix', x=0.5):
        """
        Compute an interpolated point between two geometry messages or tf2 frames.

        Args:
            point_or_pose_a (str | Point | PointStamped | Pose | PoseStamped): First point or pose. String values are interpreted as ft2 frames.
            point_or_pose_b (str | Point | PointStamped | Pose | PoseStamped): Second point or pose. Same rules as above.
            mode (str, optional): Interpolation mode. 'mix' means weighted average by `x`. 'distance' means step `x` meters from B toward A. Defaults to 'mix'.
            x (float | int, optional): Mix ratio (0–1) or distance in meters depending on mode. Defaults to 0.5.

        Returns:
            tuple[bool, str, PointStamped | None]:
                - Success flag.
                - Human-readable message.
                - Interpolated point if successful.
        """
        # parse arguments
        assert_type_value(obj=point_or_pose_a, type_or_value=[str, Point, PointStamped, Pose, PoseStamped], name="argument 'point_or_pose_a'", logger=self._logger)
        assert_type_value(obj=point_or_pose_b, type_or_value=[str, Point, PointStamped, Pose, PoseStamped], name="argument 'point_or_pose_b'", logger=self._logger)
        assert_type_value(obj=mode, type_or_value=["mix", "distance"], name="argument 'mode'", logger=self._logger)
        assert_type_value(obj=x, type_or_value=[float, int], name="argument 'x'", logger=self._logger)
        if not x >= 0.0:
            message = f"Expected value of argument 'x' to be greater than or equal to zero, but is '{x}'."
            self._logger.error(message)
            assert x >= 0.0, message
        if mode == "mix":
            if not x <= 1.0:
                message = f"Expected value of argument 'x' to be less than or equal to one with mode '{mode}', but is '{x}'."
                self._logger.error(message)
                assert x <= 1.0, message

        # convert string inputs to PointStamped
        if isinstance(point_or_pose_a, str):
            point_or_pose_a = create_point(point=[0, 0, 0], frame=point_or_pose_a, stamp=self._node.get_clock().now())
        if isinstance(point_or_pose_b, str):
            point_or_pose_b = create_point(point=[0, 0, 0], frame=point_or_pose_b, stamp=self._node.get_clock().now())

        # transform inputs to same frame, if both provide one and differ
        is_stamped_a = isinstance(point_or_pose_a, PointStamped) or isinstance(point_or_pose_a, PoseStamped)
        is_stamped_b = isinstance(point_or_pose_b, PointStamped) or isinstance(point_or_pose_b, PoseStamped)
        if is_stamped_a and is_stamped_b:
            success, message, point_or_pose_b = self.transform_to_frame(point_or_pose_b, point_or_pose_a.header.frame_id)
            if not success:
                message = f"Failed in transform_to_frame(): {message}"
                return False, message, None
            else:
                message_prefix = ""
        else:
            message_prefix = "Assuming frames of two non-stamped points matches. "
            self._logger.warn(message_prefix[:-1])

        # extract points
        if isinstance(point_or_pose_a, Point):
            a_np = np.array([point_or_pose_a.x, point_or_pose_a.y, point_or_pose_a.z])
        elif isinstance(point_or_pose_a, PointStamped):
            a_np = np.array([point_or_pose_a.point.x, point_or_pose_a.point.y, point_or_pose_a.point.z])
        elif isinstance(point_or_pose_a, Pose):
            a_np = np.array([point_or_pose_a.position.x, point_or_pose_a.position.y, point_or_pose_a.position.z])
        elif isinstance(point_or_pose_a, PoseStamped):
            a_np = np.array([point_or_pose_a.pose.position.x, point_or_pose_a.pose.position.y, point_or_pose_a.pose.position.z])
        if isinstance(point_or_pose_b, Point):
            b_np = np.array([point_or_pose_b.x, point_or_pose_b.y, point_or_pose_b.z])
        elif isinstance(point_or_pose_b, PointStamped):
            b_np = np.array([point_or_pose_b.point.x, point_or_pose_b.point.y, point_or_pose_b.point.z])
        elif isinstance(point_or_pose_b, Pose):
            b_np = np.array([point_or_pose_b.position.x, point_or_pose_b.position.y, point_or_pose_b.position.z])
        elif isinstance(point_or_pose_b, PoseStamped):
            b_np = np.array([point_or_pose_b.pose.position.x, point_or_pose_b.pose.position.y, point_or_pose_b.pose.position.z])

        # interpolate
        if mode == "distance":
            d = np.linalg.norm(a_np - b_np)
            if d < x:
                message = f"The distance between the provided points is '{d:.3f}m', which is smaller than the specified distance '{x:.3f}m'."
                self._logger.error(message)
                return False, message, None
            x = x / d
            result_np = b_np + (a_np - b_np) * x
        elif mode == "mix":
            result_np = a_np + (b_np - a_np) * x

        # create interpolated point
        result = PointStamped()
        result.header.frame_id = point_or_pose_a.header.frame_id
        result.header.stamp = point_or_pose_b.header.stamp
        result.point.x = result_np[0]
        result.point.y = result_np[1]
        result.point.z = result_np[2]

        # log
        message = f"Successfully interpolated between two points using mode '{mode}' with parameter '{x}'."
        self._logger.debug(message)

        return True, message, result

    def get_distance(self, point_or_pose_a, point_or_pose_b):
        """
        Compute the Euclidean distance between two points or poses in 3D.

        Args:
            point_or_pose_a (str | Point | PointStamped | Pose | PoseStamped): First input geometry or tf2 frame.
            point_or_pose_b (str | Point | PointStamped | Pose | PoseStamped): Second input geometry or tf2 frame.

        Returns:
            tuple[bool, str, float | None]:
                - Success flag.
                - Human-readable message.
                - Distance in meters if successful.

        Notes:
            - Non-stamped inputs are assumed to be in the same frame.
        """
        # parse arguments
        assert_type_value(obj=point_or_pose_a, type_or_value=[str, Point, PointStamped, Pose, PoseStamped], name="argument 'point_or_pose_a'", logger=self._logger)
        assert_type_value(obj=point_or_pose_b, type_or_value=[str, Point, PointStamped, Pose, PoseStamped], name="argument 'point_or_pose_b'", logger=self._logger)

        # convert string inputs to PointStamped
        if isinstance(point_or_pose_a, str):
            point_or_pose_a = create_point(point=[0, 0, 0], frame=point_or_pose_a, stamp=self._node.get_clock().now())
        if isinstance(point_or_pose_b, str):
            point_or_pose_b = create_point(point=[0, 0, 0], frame=point_or_pose_b, stamp=self._node.get_clock().now())

        # transform inputs to same frame, if both provide one and differ
        is_stamped_a = isinstance(point_or_pose_a, PointStamped) or isinstance(point_or_pose_a, PoseStamped)
        is_stamped_b = isinstance(point_or_pose_b, PointStamped) or isinstance(point_or_pose_b, PoseStamped)
        if is_stamped_a and is_stamped_b:
            success, message, point_or_pose_b = self.transform_to_frame(point_or_pose_b, point_or_pose_a.header.frame_id)
            if not success:
                message = f"Failed in transform_to_frame(): {message}"
                return False, message, None
            else:
                message_prefix = ""
        else:
            message_prefix = "Assuming frames of two non-stamped points matches. "
            self._logger.warn(message_prefix[:-1])

        # extract points
        if isinstance(point_or_pose_a, Point):
            a_np = np.array([point_or_pose_a.x, point_or_pose_a.y, point_or_pose_a.z])
        elif isinstance(point_or_pose_a, PointStamped):
            a_np = np.array([point_or_pose_a.point.x, point_or_pose_a.point.y, point_or_pose_a.point.z])
        elif isinstance(point_or_pose_a, Pose):
            a_np = np.array([point_or_pose_a.position.x, point_or_pose_a.position.y, point_or_pose_a.position.z])
        elif isinstance(point_or_pose_a, PoseStamped):
            a_np = np.array([point_or_pose_a.pose.position.x, point_or_pose_a.pose.position.y, point_or_pose_a.pose.position.z])
        if isinstance(point_or_pose_b, Point):
            b_np = np.array([point_or_pose_b.x, point_or_pose_b.y, point_or_pose_b.z])
        elif isinstance(point_or_pose_b, PointStamped):
            b_np = np.array([point_or_pose_b.point.x, point_or_pose_b.point.y, point_or_pose_b.point.z])
        elif isinstance(point_or_pose_b, Pose):
            b_np = np.array([point_or_pose_b.position.x, point_or_pose_b.position.y, point_or_pose_b.position.z])
        elif isinstance(point_or_pose_b, PoseStamped):
            b_np = np.array([point_or_pose_b.pose.position.x, point_or_pose_b.pose.position.y, point_or_pose_b.pose.position.z])

        # compute distance
        distance = np.linalg.norm(a_np - b_np)

        # log
        message = f"The distance between the two points is '{distance:.3f}m'."
        self._logger.debug(message)

        return True, f"{message_prefix}{message}", distance
