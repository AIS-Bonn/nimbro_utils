#!/usr/bin/env python3

import copy
import time
import threading

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import CameraInfo, Image, CompressedImage

from nimbro_utils.node_extensions.logger import Logger
from nimbro_utils.utility.image import convert_image
from nimbro_utils.utility.node import create_throttled_subscription
from nimbro_utils.utility.misc import assert_type_value, assert_keys, update_dict

default_settings = {
    # Logger severity in [10, 20, 30, 40, 50] (int).
    'severity': 20,
    # Logger suffix (str).
    'suffix': "depth_converter",
    # Name of the CameraInfo topic used for depth conversion (str).
    'depth_info_topic': None, # "/gemini/depth/camera_info",
    # Set None to only receive CameraInfo once with out updates or throttle it by providing the minimum interval in seconds >= 0.0 (int | float).
    'depth_info_update': None,
    # Scale the depth to match target units (float).
    'depth_scale': 0.001
}

class DepthConverter:
    """
    A utility to project pixel coordinates of a depth image to a pointcloud,
    using the newest CameraInfo corresponding to the depth image,
    and supporting various camera models.
    """

    def __init__(self, node, settings=None):
        """
        Initialize the DepthConverter.

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

        # state
        self._lock = threading.Lock()

        # settings
        settings = update_dict(old_dict=default_settings, new_dict=settings)
        self.set_settings(settings=settings, keep_existing=False)

    def _camera_params_callback(self, msg):
        self._lock.acquire()
        self._logger.debug("Receiving camera parameters")

        depth_info = {
            "frame": msg.header.frame_id,
            "model": msg.distortion_model,
            "width": msg.width,
            "height": msg.height,
            "focal_x": msg.p[0],
            "focal_y": msg.p[5],
            "center_x": msg.p[2],
            "center_y": msg.p[6],
        }

        if self._depth_info == depth_info:
            self._logger.debug("Ignoring unchanged camera parameters")
            if self._settings['depth_info_update'] is None:
                self._node.destroy_subscription(self._sub_camera_info)
                self._logger.debug("Destroyed CameraInfo subscriber")
            self._lock.release()
            return

        if self._depth_info is None:
            init = True
        else:
            init = False

        self._depth_info = update_dict(
            old_dict={} if init else self._depth_info,
            new_dict=depth_info,
            key_name="depth info",
            logger=self._logger,
            info=False,
            debug=False
        )

        if self._depth_info['model'] == "plumb_bob":
            model = PlumbBob.from_camera_info_message(msg)
        elif self._depth_info['model'] == "rational_polynomial":
            model = RationalPolynomial.from_camera_info_message(msg)
        elif self._depth_info['model'] == "double_sphere":
            model = DoubleSphere.from_camera_info_message(msg)
        else:
            raise NotImplementedError(f"Unknown camera model '{self._depth_info['model']}'")

        self._precompute_depth(model)

        self._logger.info(f"{'Initialized' if init else 'Updated'} camera parameters")

        if self._settings['depth_info_update'] is None:
            self._node.destroy_subscription(self._sub_camera_info)
            self._logger.debug("Destroyed CameraInfo subscriber")
        self._lock.release()

    def _precompute_depth(self, model):
        coords_u, coords_v = np.meshgrid(np.arange(self._depth_info['width']), np.arange(self._depth_info['height']))
        coords_uv = np.stack((coords_u, coords_v), axis=0).reshape((2, -1))
        self._full_pixel_mask = coords_uv.transpose(1, 0)

        # add batch dimension, just for pytorch-like style. Remove it afterwards.
        coords_uv = coords_uv[None, ...]
        coords_xyz, mask_valid = model.project_image_onto_points(coords_uv)
        coords_xyz = coords_xyz[0]
        mask_valid = mask_valid[0]

        # normalize by z component
        coords_xyz = np.divide(coords_xyz, coords_xyz[2, :], where=coords_xyz[2, :] > 0)

        coords_xyz[:, ~mask_valid] = 0.0
        self._vectors_depth_1 = coords_xyz.transpose(1, 0).reshape((self._depth_info['height'], self._depth_info['width'], -1))

    def _assert_camera_info(self):
        if self._depth_info is None:
            message = f"Cannot convert depth before receiving camera info on topic '{self._settings['depth_info_topic']}'."
            self._logger.error(message)
            assert self._depth_info is not None, message

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

        Notes:
            Triggering CameraInfo update once is possible by explicitly including 'depth_info_update' in settings and setting it to None.
        """
        # parse arguments
        assert_type_value(obj=settings, type_or_value=dict, name="argument 'settings'", logger=self._logger)
        if hasattr(self, "_settings") and 'depth_info_update' in settings and settings.get('depth_info_update') is None and self._settings['depth_info_update'] is None:
            self._logger.debug("Explicitly forcing update of CameraInfo")
            recreate_sub = True
        else:
            recreate_sub = False
        assert_type_value(obj=keep_existing, type_or_value=bool, name="argument 'keep_existing'", logger=self._logger)
        settings = update_dict(old_dict=self._settings if keep_existing else {}, new_dict=settings, key_name="setting", logger=self._logger, info=False, debug=False)
        assert_keys(obj=settings, keys=default_settings.keys(), mode="match", name="settings", logger=self._logger)

        # Logger
        self._logger.set_settings({'severity': settings['severity'], 'name': settings['suffix']})

        # depth_scale
        assert_type_value(obj=settings['depth_scale'], type_or_value=float, name="setting 'depth_scale'", logger=self._logger)

        # CameraInfo
        assert_type_value(obj=settings['depth_info_update'], type_or_value=[None, int, float], name="setting 'depth_info_update'", logger=self._logger)
        assert_type_value(obj=settings['depth_info_topic'], type_or_value=str, name="setting 'depth_info_topic'", logger=self._logger)
        self._lock.acquire()
        if hasattr(self, "_settings"):
            if (self._settings['depth_info_update'] != settings['depth_info_update'] and settings['depth_info_update'] is not None) or self._settings['depth_info_topic'] != settings['depth_info_topic'] or recreate_sub:
                if self._settings['depth_info_topic'] != settings['depth_info_topic']:
                    self._depth_info = None
                    self._depth_image = None
                set_sub = True
                if self._node.destroy_subscription(self._sub_camera_info):
                    self._logger.debug("Destroyed CameraInfo subscriber")
            else:
                set_sub = False
        else:
            self._depth_info = None
            self._depth_image = None
            set_sub = True
        if set_sub:
            qos_profile = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
            if settings['depth_info_update'] is None:
                self._logger.debug("Creating CameraInfo subscriber")
                self._sub_camera_info = self._node.create_subscription(CameraInfo, settings['depth_info_topic'], self._camera_params_callback, qos_profile, callback_group=MutuallyExclusiveCallbackGroup())
            else:
                self._logger.debug("Creating throttled CameraInfo subscriber")
                self._sub_camera_info = create_throttled_subscription(self._node, CameraInfo, settings['depth_info_topic'], self._camera_params_callback, qos_profile, throttle=settings['depth_info_update'], callback_group=MutuallyExclusiveCallbackGroup())

        self._settings = settings
        self._lock.release()

    # CameraInfo

    def wait_for_camera(self):
        """
        Wait until CameraInfo has been received on topic set by 'depth_info_topic'.
        """
        while not self.is_camera_received():
            self._logger.info(f"Waiting for CameraInfo on topic '{self._settings['depth_info_topic']}'", skip_first=True, throttle_duration_sec=1.0)
            time.sleep(0.1)

    def is_camera_received(self):
        """
        Check if CameraInfo has been received on topic set by 'depth_info_topic'.

        Returns:
            bool: True if CameraInfo has been received.
        """
        return self._depth_info is not None

    def get_camera(self):
        """
        Retrieve the current CameraInfo.

        Raises:
            AssertionError: If CameraInfo has not been received yet.

        Returns:
            dict: A deep copy of the latest CameraInfo parameters received on topic set by 'depth_info_topic'.
        """
        self._assert_camera_info()
        return copy.deepcopy(self._depth_info)

    def get_pixels(self):
        """
        Retrieve a list of all pixel coordinates with respect to the set CameraInfo.

        Raises:
            AssertionError: If CameraInfo has not been received yet.

        Returns:
            numpy.ndarray: Array of shape (N, 2) with all image coordinates (x, y) of the depth image described on topic set by 'depth_info_topic'.
        """
        self._assert_camera_info()
        return self._full_pixel_mask.copy()

    # convert depth to pointcloud

    def set_depth(self, depth_msg):
        """
        Set the depth image explicitly to omit redundant computations,
        when retrieving multiple point clouds from a single depth image,
        by using 'get_point_cloud()' and setting 'depth_msg' to None.

        Args:
            depth_msg (Image | CompressedImage): A depth message corresponding to the CameraInfo received on topic set by 'depth_info_topic'.

        Raises:
            AssertionError: If input arguments or provided depth message are invalid, or if CameraInfo has not been received yet.
        """
        self._assert_camera_info()
        assert_type_value(obj=depth_msg, type_or_value=[Image, CompressedImage], name="argument 'depth_msg'", logger=self._logger)
        if depth_msg.header.frame_id == "":
            self._logger.warn(f"Received depth image with empty frame. Assuming it matches the expected frame '{self._depth_info['frame']}'", throttle_duration_sec=self._settings['throttle'])
        elif not depth_msg.header.frame_id == self._depth_info['frame']:
            message = f"Received depth image with frame '{depth_msg.header.frame_id}' but expected frame to be '{self._depth_info['frame']}'."
            self._logger.error(message)
            assert depth_msg.header.frame_id == self._depth_info['frame'], message

        if isinstance(depth_msg, Image):
            self._lock.acquire()
            self._depth_image = convert_image(image=depth_msg, target_format="NumPy", target_encoding="16UC1", logger=self._logger)
            self._depth_image = np.array(self._depth_image * self._settings['depth_scale'], dtype=np.float32)
            if self._settings['severity'] == 10:
                self._logger.debug(f"Set Image: msg_encoding={depth_msg.encoding} cv_nanmax={np.nanmax(self._depth_image)} cv_nanmin={np.nanmin(self._depth_image)} cv_dtype={self._depth_image.dtype}")
            if not self._depth_image.shape == (self._depth_info['height'], self._depth_info['width']):
                message = f"Expected shape of depth image '{self._depth_image.shape}' to match CameraInfo '{(self._depth_info['height'], self._depth_info['width'])}'."
                self._logger.error(message)
                assert self._depth_image.shape == (self._depth_info['height'], self._depth_info['width']), message
            self._lock.release()
        elif isinstance(depth_msg, CompressedImage):
            self._lock.acquire()
            self._depth_image = convert_image(image=depth_msg, target_format="NumPy", target_encoding="16UC1", logger=self._logger)
            self._depth_image = np.array(self._depth_image * self._settings['depth_scale'], dtype=np.float32)
            if self._settings['severity'] == 10:
                self._logger.debug(f"Set CompressedImage: msg_format={depth_msg.format} cv_nanmax={np.nanmax(self._depth_image)} cv_nanmin={np.nanmin(self._depth_image)} cv_dtype={self._depth_image.dtype}")
                if not self._depth_image.shape == (self._depth_info['height'], self._depth_info['width']):
                    message = f"Expected shape of depth image '{self._depth_image.shape}' to match CameraInfo '{(self._depth_info['height'], self._depth_info['width'])}'."
                    self._logger.error(message)
                    assert self._depth_image.shape == (self._depth_info['height'], self._depth_info['width']), message
            self._lock.release()
        else:
            raise NotImplementedError(f"Type of depth_msg: {type(depth_msg).__name__}")

    def is_depth_set(self):
        """
        Check if a depth image is currently set by `set_depth()` or `set_depth()`.

        Returns:
            bool: True depth image is set.
        """
        return self._depth_image is not None

    def get_point_cloud(self, pixels=None, depth_msg=None, filter_invalid=True):
        """
        Returns 3D points in the camera frame.

        Args:
            pixels (list | numpy.ndarray | None, optional): Array of shape (N, 2) with image coordinates (x, y) that are to be converted to 3D points.
                                                            Pass None to select all pixels. Defaults to None.
            depth_msg (Image | CompressedImage, optional): A depth message corresponding to the CameraInfo received on topic set by 'depth_info_topic'. Defaults to None.
            filter_invalid (bool, optional): If True, invalid values (nan, +inf, -inf, 0) are removed from the result.
                                             If False invalid values are mapped to 'numpy.nan'. Defaults to True.
        Raises:
            AssertionError: If CameraInfo has not been received yet.

        Returns:
            numpy.ndarray: Array of shape (N, 3) with points (x, y, z) ordered according to pixels.
        """
        # parse arguments

        self._assert_camera_info()
        assert_type_value(obj=filter_invalid, type_or_value=bool, name="argument 'filter_invalid'", logger=self._logger)

        assert_type_value(obj=pixels, type_or_value=[np.ndarray, list, None], name="argument 'pixels'", logger=self._logger)
        if pixels is not None:
            if isinstance(pixels, list):
                pixels = np.array(pixels, dtype=np.int32)
            if not len(pixels.shape) == 2:
                message = f"Expected 'pixels' to have shape '(N, 2)' but it is '{pixels.shape}'."
                self._logger.error(message)
                assert len(pixels.shape) == 2, message
            if not pixels.shape[1] == 2:
                message = f"Expected 'pixels' to have shape '(N, 2)' but it is '{pixels.shape}'."
                self._logger.error(message)
                assert pixels.shape[1] == 2, message

        assert_type_value(obj=depth_msg, type_or_value=[Image, CompressedImage, None], name="argument 'depth_msg'", logger=self._logger)
        if depth_msg is None:
            assert_type_value(obj=self._depth_image, type_or_value=np.ndarray, name="attribute '_depth_image'", text="Use set_depth() before get_point_cloud() or provide a depth message.", logger=self._logger)
        else:
            self.set_depth(depth_msg)

        # project pixels
        # see examples/depth_converter_shootout.py for comparing algorithms. Actually, one could find some number of pixels N above which the upper algorithm is preferable.

        self._lock.acquire()

        if pixels is None:
            coords = self._full_pixel_mask[:, 1], self._full_pixel_mask[:, 0]
            Z = self._depth_image[coords]
            Z_mask = np.logical_and(np.isfinite(Z), Z != 0)
            if filter_invalid:
                Z = Z[Z_mask]
                X = (coords[1][Z_mask] - self._depth_info['center_x']) * Z / self._depth_info['focal_x']
                Y = (coords[0][Z_mask] - self._depth_info['center_y']) * Z / self._depth_info['focal_y']
            else:
                Z[~Z_mask] = np.nan
                X = (coords[1] - self._depth_info['center_x']) * Z / self._depth_info['focal_x']
                Y = (coords[0] - self._depth_info['center_y']) * Z / self._depth_info['focal_y']
            cloud = np.stack([X, Y, Z], axis=-1)
        else:
            coords = pixels[:, 1], pixels[:, 0]
            Z = self._depth_image[coords]
            Z_mask = np.logical_and(np.isfinite(Z), Z != 0)
            if filter_invalid:
                Z = Z[Z_mask]
                selected_vectors = self._vectors_depth_1[coords[0][Z_mask], coords[1][Z_mask], :]
            else:
                Z[~Z_mask] = np.nan
                selected_vectors = self._vectors_depth_1[coords[0], coords[1], :]
            cloud = selected_vectors * Z[:, np.newaxis]

        self._lock.release()

        return cloud

# camera models

class CameraModel:
    """Base class for camera models."""

    def __init__(self, fx, fy, cx, cy, model_distortion, params_distortion, shape_image):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.model_distortion = model_distortion
        self.params_distortion = params_distortion
        self.shape_image = shape_image

    @classmethod
    def from_camera_info_message(cls, message):
        """Create an instance from a camera info message."""
        try:
            binning_x = message.binning_x if message.binning_x != 0 else 1
            binning_y = message.binning_y if message.binning_y != 0 else 1
        except AttributeError:
            binning_x = 1
            binning_y = 1

        try:
            offset_x = message.roi.offset_x
            offset_y = message.roi.offset_y
            # Do not know channel dimension from camera info message but keep it for pytorch-like style
            shape_image = (-1, message.roi.height, message.roi.width)
        except AttributeError:
            offset_x = 0
            offset_y = 0
            shape_image = (-1, message.height, message.width)

        fx = message.k[0] / binning_x
        fy = message.k[4] / binning_y
        cx = (message.k[2] - offset_x) / binning_x
        cy = (message.k[5] - offset_y) / binning_y

        model_distortion = message.distortion_model
        params_distortion = cls.create_dict_params_distortion(message.d)

        instance = cls(fx, fy, cx, cy, model_distortion, params_distortion, shape_image)
        return instance

    @classmethod
    def create_dict_params_distortion(cls, list_params_distortion):
        try:
            params_distortion = dict(zip(cls.keys_params_distortion, list_params_distortion))
        except AttributeError:
            params_distortion = dict(enumerate(list_params_distortion))

        return params_distortion

    def project_points_onto_image(self, coords_xyz):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        raise NotImplementedError()

    def project_image_onto_points(self, coords_uv):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        raise NotImplementedError()

class PlumbBob(CameraModel):
    def project_points_onto_image(self, coords_xyz):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        coords_uv = np.stack((u, v), axis=1)

        mask_valid = np.ones_like(u, dtype=bool)

        return coords_uv, mask_valid

    def project_image_onto_points(self, coords_uv):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        mz = np.ones_like(mx)

        factor = 1.0 / np.sqrt(mx**2 + my**2 + 1.0)
        coords_xyz = factor[:, None, :] * np.stack((mx, my, mz), axis=1)

        mask_valid = np.ones_like(mx, dtype=bool)

        return coords_xyz, mask_valid

class RationalPolynomial(CameraModel):
    # Note: Distortion is ignored for now since they do not have a noticeable impact for the gemini.

    keys_params_distortion = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def project_points_onto_image(self, coords_xyz):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        coords_uv = np.stack((u, v), axis=1)

        mask_valid = np.ones_like(u, dtype=bool)

        return coords_uv, mask_valid

    def project_image_onto_points(self, coords_uv):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        mz = np.ones_like(mx)

        factor = 1.0 / np.sqrt(mx**2 + my**2 + 1.0)
        coords_xyz = factor[:, None, :] * np.stack((mx, my, mz), axis=1)

        mask_valid = np.ones_like(mx, dtype=bool)

        return coords_xyz, mask_valid

class DoubleSphere(CameraModel):
    """Implemented according to:
    V. Usenko, N. Demmel, and D. Cremers: The Double Sphere Camera Model.
    Proceedings of the International Conference on 3D Vision (3DV) (2018).
    URL: https://arxiv.org/pdf/1807.08957.pdf."""

    keys_params_distortion = ["xi", "alpha"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = self.params_distortion["alpha"]
        self.xi = self.params_distortion["xi"]

    def project_points_onto_image(self, coords_xyz, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_xyz = coords_xyz.astype(np.float16)

        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        # Eq. (41)
        d1 = np.sqrt(x**2 + y**2 + z**2)
        # Eq. (45)
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        # Eq. (44)
        w2 = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi**2 + 1.0)
        # Eq. (43)
        mask_valid = z > -w2 * d1

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            x = x[mask_valid][None, ...]
            y = y[mask_valid][None, ...]
            z = z[mask_valid][None, ...]
            d1 = d1[mask_valid][None, ...]
            mask_valid = np.ones_like(z, dtype=bool)

        # Eq. (42)
        z_shifted = self.xi * d1 + z
        d2 = np.sqrt(x**2 + y**2 + z_shifted**2)
        # Eq. (40)
        denominator = self.alpha * d2 + (1 - self.alpha) * z_shifted
        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy
        coords_uv = np.stack([u, v], axis=1)

        if use_mask_fov:
            mask_left = coords_uv[:, 0, :] >= 0
            mask_top = coords_uv[:, 1, :] >= 0
            mask_right = coords_uv[:, 0, :] < self.shape_image[2]
            mask_bottom = coords_uv[:, 1, :] < self.shape_image[1]
            mask_valid *= mask_left * mask_top * mask_right * mask_bottom

        return coords_uv, mask_valid

    def project_image_onto_points(self, coords_uv, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_uv = coords_uv.astype(np.float16)

        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        # Eq. (47)
        mx = (u - self.cx) / self.fx
        # Eq. (48)
        my = (v - self.cy) / self.fy
        # Eq. (49)
        square_r = mx**2 + my**2
        # Eq. (51) can be written to use this
        term = 1.0 - (2.0 * self.alpha - 1.0) * square_r
        # Eq. (51)
        mask_valid = term >= 0.0 if self.alpha > 0.5 else np.ones_like(term, dtype=bool)

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            mx = mx[mask_valid][None, ...]
            my = my[mask_valid][None, ...]
            square_r = square_r[mask_valid][None, ...]
            term = term[mask_valid][None, ...]
            mask_valid = np.ones_like(term, dtype=bool)

        # Eq. (50)
        mz = (1.0 - self.alpha**2 * square_r) / (self.alpha * np.sqrt(term) + 1.0 - self.alpha)
        # Eq. (46)
        factor = (mz * self.xi + np.sqrt(mz**2 + (1.0 - self.xi**2) * square_r)) / (mz**2 + square_r)
        coords_xyz = factor[:, None, :] * np.stack((mx, my, mz), axis=1)
        coords_xyz[:, 2, :] -= self.xi

        if use_mask_fov:
            mask_behind = coords_xyz[:, 2, :] > 0.0
            mask_valid *= mask_behind

        return coords_xyz, mask_valid
