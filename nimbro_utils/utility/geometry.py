#!/usr/bin/env python3

import struct
from ctypes import cast, pointer, c_float, POINTER, c_uint32

import numpy as np

import builtin_interfaces.msg
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan, PointField, PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf2_geometry_msgs import do_transform_point, do_transform_pose_stamped
from geometry_msgs.msg import Point, PointStamped, Vector3, Vector3Stamped
from geometry_msgs.msg import Quaternion, QuaternionStamped, Pose, PoseStamped, Transform, TransformStamped

from nimbro_utils.utility.misc import assert_type_value, convert_stamp

# convert between rotation formalisms

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion into a 3×3 rotation matrix.

    Args:
        quaternion (Quaternion | numpy.ndarray | list | tuple): Quaternion with non-zero norm ([qx, qy, qz, qw] when using array type).

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        numpy.ndarray: 3×3 rotation matrix.
    """
    # parse arguments
    assert_type_value(obj=quaternion, type_or_value=[Quaternion, np.ndarray, list, tuple], name="argument 'quaternion'")
    if isinstance(quaternion, Quaternion):
        quaternion = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    elif isinstance(quaternion, np.ndarray):
        assert quaternion.shape == (4,), f"Expected 'quaternion' to have shape '(4,)' (qx, qy, qz, qw) but got '{quaternion.shape}'."
    else:
        assert len(quaternion) == 4, f"Expected 'quaternion' to have '4' elements (qx, qy, qz, qw) but got '{len(quaternion)}'."
    norm = np.linalg.norm(quaternion)
    assert norm > 0, "Expected 'quaternion' to have non-zero norm but got zero."

    # normalize
    quaternion = np.asarray(quaternion, dtype=float) / norm

    # compute the rotation matrix
    qx, qy, qz, qw = quaternion
    rotation_matrix = np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])

    return rotation_matrix

def quaternion_to_yaw_pitch_roll(quaternion, degrees=False):
    """
    Convert a quaternion into Euler angles (yaw, pitch, roll).

    Args:
        quaternion (Quaternion | numpy.ndarray | list | tuple): Quaternion with non-zero norm ([qx, qy, qz, qw] when using array type).
        degrees (bool, optional): If True, return angles in degrees. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        numpy.ndarray: Euler angles (yaw, pitch, roll) in specified units.
    """
    # parse arguments
    assert_type_value(obj=quaternion, type_or_value=[Quaternion, np.ndarray, list, tuple], name="argument 'quaternion'")
    if isinstance(quaternion, Quaternion):
        quaternion = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    elif isinstance(quaternion, np.ndarray):
        assert quaternion.shape == (4,), f"Expected 'quaternion' to have shape '(4,)' (qx, qy, qz, qw) but got '{quaternion.shape}'."
    else:
        assert len(quaternion) == 4, f"Expected 'quaternion' to have '4' elements (qx, qy, qz, qw) but got '{len(quaternion)}'."
    norm = np.linalg.norm(quaternion)
    assert norm > 0, "Expected 'quaternion' to have non-zero norm but got zero."
    assert_type_value(obj=degrees, type_or_value=bool, name="argument 'degrees'")

    # normalize
    quaternion = np.asarray(quaternion, dtype=float) / norm

    # convert
    x, y, z, w = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x**2 + y**2)
    roll = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(t3, t4)

    # format
    ypr = np.array([yaw, pitch, roll])
    if degrees:
        ypr = ypr * 180.0 / np.pi

    return ypr

def yaw_pitch_roll_to_rotation_matrix(yaw_pitch_roll, degrees=False):
    """
    Convert Euler angles (yaw, pitch, roll) into a rotation matrix.

    Args:
        yaw_pitch_roll (numpy.ndarray | list | tuple): Euler angles (yaw, pitch, roll).
        degrees (bool, optional): If True, interpret `yaw_pitch_roll` as degrees instead of radians. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        numpy.ndarray: 3×3 rotation matrix.
    """
    # parse arguments
    assert_type_value(obj=yaw_pitch_roll, type_or_value=[np.ndarray, list, tuple], name="argument 'yaw_pitch_roll'")
    angles = np.asarray(yaw_pitch_roll, dtype=float)
    assert angles.shape == (3,), f"Expected 'yaw_pitch_roll' to have shape '(3,)' but got '{angles.shape}'."
    assert_type_value(obj=degrees, type_or_value=bool, name="argument 'degrees'")

    # adjust units
    if degrees:
        angles = angles * np.pi / 180.0
    yaw, pitch, roll = angles

    # compute matrices
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    return R_z @ R_y @ R_x

def yaw_pitch_roll_to_quaternion(yaw_pitch_roll, degrees=False):
    """
    Convert Euler angles (yaw, pitch, roll) into a quaternion.

    Args:
        yaw_pitch_roll (numpy.ndarray | list | tuple): Euler angles (yaw, pitch, roll).
        degrees (bool, optional): If True, interpret `yaw_pitch_roll` as degrees instead of radians. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        numpy.ndarray: Unit quaternion [qx, qy, qz, qw].
    """
    # parse arguments
    assert_type_value(obj=yaw_pitch_roll, type_or_value=[np.ndarray, list, tuple], name="argument 'yaw_pitch_roll'")
    angles = np.asarray(yaw_pitch_roll, dtype=float)
    assert angles.shape == (3,), f"Expected 'yaw_pitch_roll' to have shape '(3,)' but got '{angles.shape}'."
    assert_type_value(obj=degrees, type_or_value=bool, name="argument 'degrees'")

    # adjust units
    if degrees:
        angles = angles * np.pi / 180.0
    yaw, pitch, roll = angles

    # compute quaternion

    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    quaternion = np.array([qx, qy, qz, qw])

    return quaternion / np.linalg.norm(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a 3×3 rotation matrix into a quaternion.

    Args:
        rotation_matrix (numpy.ndarray): 3×3 rotation matrix.

    Raises:
        AssertionError: If input arguments are invalid.
        ImportError: If `scipy` is not available.

    Returns:
        numpy.ndarray: Unit quaternion [qx, qy, qz, qw].
    """
    # parse arguments
    assert_type_value(obj=rotation_matrix, type_or_value=np.ndarray, name="argument 'rotation_matrix'")
    assert rotation_matrix.shape == (3, 3), f"Expected 'rotation_matrix' to have shape '(3,3)' but got '{rotation_matrix.shape}'."

    # convert
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_matrix(rotation_matrix)
    quat = rotation.as_quat()

    return quat

# convert between NumPy and ROS2 messages

def create_point(point, frame=None, stamp=None, strict=True):
    """
    Create a Point or PointStamped.

    Args:
        point (Point, numpy.ndarray, list, tuple): Point ([x, y, z] when using array type).
        frame (str | None, optional): If provided, returns PointStamped with this frame. Defaults to None.
        stamp (see convert_stamp() | None, optional): If provided, returns PointStamped with this time stamp. Defaults to None.
        strict (bool, optional): If True, asserts that frame is not empty and stamp is not zero, if at least one of the two is not None. Defaults to True.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        Point | PointStamped: Created point.
    """
    # parse arguments
    assert_type_value(obj=point, type_or_value=[Point, np.ndarray, list, tuple], name="argument 'point'")
    if isinstance(point, np.ndarray):
        point = point.tolist()
    if not isinstance(point, Point):
        assert len(point) == 3, f"Expected 'point' to have '3' elements (x, y, z) but got '{len(point)}'."
        for item in point:
            assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'point'")
    assert_type_value(obj=frame, type_or_value=[str, None], name="argument 'frame'")
    assert_type_value(obj=strict, type_or_value=bool, name="argument 'strict'")
    if stamp is not None:
        stamp_msg = convert_stamp(stamp=stamp, target_format="msg")
    if strict:
        if frame is not None:
            assert frame != "", "Expected 'frame' to either be None or a non-empty string."
            assert stamp is not None, "Expected 'stamp' to not be None when frame is provided."
        if stamp is not None:
            assert stamp_msg != builtin_interfaces.msg.Time(), "Expected 'stamp' to either be None or Unix epoch time greater zero."
            assert frame is not None, "Expected 'stamp' to not be None when stamp is provided."

    # create point
    if isinstance(point, Point):
        point_msg = point
    else:
        point_msg = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
    if frame is None and stamp is None:
        return point_msg
    else:
        point_stamped = PointStamped()
        point_stamped.point = point_msg
        if frame is not None:
            point_stamped.header.frame_id = frame
        if stamp is not None:
            point_stamped.header.stamp = stamp_msg
        return point_stamped

def create_vector(vector, frame=None, stamp=None, strict=True):
    """
    Create a Vector3 or Vector3Stamped.

    Args:
        vector (Vector3, numpy.ndarray, list, tuple): Vector3 ([x, y, z] when using array type).
        frame (str | None, optional): If provided, returns Vector3Stamped with this frame. Defaults to None.
        stamp (see convert_stamp() | None, optional): If provided, returns Vector3Stamped with this time stamp. Defaults to None.
        strict (bool, optional): If True, asserts that frame is not empty and stamp is not zero, if at least one of the two is not None. Defaults to True.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        Vector3 | Vector3Stamped: Created vector.
    """
    # parse arguments
    assert_type_value(obj=vector, type_or_value=[Vector3, np.ndarray, list, tuple], name="argument 'vector'")
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    if not isinstance(vector, Vector3):
        assert len(vector) == 3, f"Expected 'vector' to have '3' elements (x, y, z) but got '{len(vector)}'."
        for item in vector:
            assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'vector'")
    assert_type_value(obj=frame, type_or_value=[str, None], name="argument 'frame'")
    assert_type_value(obj=strict, type_or_value=bool, name="argument 'strict'")
    if stamp is not None:
        stamp_msg = convert_stamp(stamp=stamp, target_format="msg")
    if strict:
        if frame is not None:
            assert frame != "", "Expected 'frame' to either be None or a non-empty string."
            assert stamp is not None, "Expected 'stamp' to not be None when frame is provided."
        if stamp is not None:
            assert stamp_msg != builtin_interfaces.msg.Time(), "Expected 'stamp' to either be None or Unix epoch time greater zero."
            assert frame is not None, "Expected 'stamp' to not be None when stamp is provided."

    # create vector
    if isinstance(vector, Vector3):
        vector_msg = vector
    else:
        vector_msg = Vector3(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))
    if frame is None and stamp is None:
        return vector_msg
    else:
        vector_stamped = Vector3Stamped()
        vector_stamped.vector = vector_msg
        if frame is not None:
            vector_stamped.header.frame_id = frame
        if stamp is not None:
            vector_stamped.header.stamp = stamp_msg
        return vector_stamped

def create_quaternion(quaternion, frame=None, stamp=None, strict=True):
    """
    Create a Quaternion or QuaternionStamped.

    Args:
        quaternion (Quaternion, numpy.ndarray, list, tuple): Quaternion ([qx, qy, qz, qw] when using array type).
        frame (str | None, optional): If provided, returns QuaternionStamped with this frame. Defaults to None.
        stamp (see convert_stamp() | None, optional): If provided, returns QuaternionStamped with this time stamp. Defaults to None.
        strict (bool, optional): If True, asserts that frame is not empty and stamp is not zero, if at least one of the two is not None. Defaults to True.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        Quaternion | QuaternionStamped: Created quaternion.
    """
    # parse arguments
    assert_type_value(obj=quaternion, type_or_value=[Quaternion, np.ndarray, list, tuple], name="argument 'quaternion'")
    if isinstance(quaternion, np.ndarray):
        quaternion = quaternion.tolist()
    if not isinstance(quaternion, Quaternion):
        assert len(quaternion) == 4, f"Expected 'quaternion' to have '3' elements (qx, qy, qz, qw) but got '{len(quaternion)}'."
        for item in quaternion:
            assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'quaternion'")
    assert_type_value(obj=frame, type_or_value=[str, None], name="argument 'frame'")
    assert_type_value(obj=strict, type_or_value=bool, name="argument 'strict'")
    if stamp is not None:
        stamp_msg = convert_stamp(stamp=stamp, target_format="msg")
    if strict:
        if frame is not None:
            assert frame != "", "Expected 'frame' to either be None or a non-empty string."
            assert stamp is not None, "Expected 'stamp' to not be None when frame is provided."
        if stamp is not None:
            assert stamp_msg != builtin_interfaces.msg.Time(), "Expected 'stamp' to either be None or Unix epoch time greater zero."
            assert frame is not None, "Expected 'stamp' to not be None when stamp is provided."

    # create quaternion
    if isinstance(quaternion, Quaternion):
        quaternion_msg = quaternion
    else:
        quaternion_msg = Quaternion(x=float(quaternion[0]), y=float(quaternion[1]), z=float(quaternion[2]), w=float(quaternion[3]))
    if frame is None and stamp is None:
        return quaternion_msg
    else:
        quaternion_stamped_msg = QuaternionStamped()
        quaternion_stamped_msg.quaternion = quaternion_msg
        if frame is not None:
            quaternion_stamped_msg.header.frame_id = frame
        if stamp is not None:
            quaternion_stamped_msg.header.stamp = stamp_msg
        return quaternion_stamped_msg

def create_pose(point, quaternion, frame=None, stamp=None, strict=True):
    """
    Create a Pose or PoseStamped.

    Args:
        point (Point, numpy.ndarray, list, tuple): Position ([x, y, z] when using array type).
        quaternion (Quaternion, numpy.ndarray, list, tuple): Quaternion ([qx, qy, qz, qw] when using array type).
        frame (str | None, optional): If provided, returns PoseStamped with this frame. Defaults to None.
        stamp (see convert_stamp() | None, optional): If provided, returns PoseStamped with this time stamp. Defaults to None.
        strict (bool, optional): If True, asserts that frame is not empty and stamp is not zero, if at least one of the two is not None. Defaults to True.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        Pose | PoseStamped: Created pose.
    """
    # parse arguments
    assert_type_value(obj=point, type_or_value=[Point, np.ndarray, list, tuple], name="argument 'point'")
    if isinstance(point, np.ndarray):
        point = point.tolist()
    if not isinstance(point, Point):
        assert len(point) == 3, f"Expected 'point' to have '3' elements (x, y, z) but got '{len(point)}'."
        for item in point:
            assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'point'")
    assert_type_value(obj=quaternion, type_or_value=[Quaternion, np.ndarray, list, tuple], name="argument 'quaternion'")
    if isinstance(quaternion, np.ndarray):
        quaternion = quaternion.tolist()
    if not isinstance(quaternion, Quaternion):
        assert len(quaternion) == 4, f"Expected 'quaternion' to have '3' elements (qx, qy, qz, qw) but got '{len(quaternion)}'."
        for item in quaternion:
            assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'quaternion'")
    assert_type_value(obj=frame, type_or_value=[str, None], name="argument 'frame'")
    assert_type_value(obj=strict, type_or_value=bool, name="argument 'strict'")
    if stamp is not None:
        stamp_msg = convert_stamp(stamp=stamp, target_format="msg")
    if strict:
        if frame is not None:
            assert frame != "", "Expected 'frame' to either be None or a non-empty string."
            assert stamp is not None, "Expected 'stamp' to not be None when frame is provided."
        if stamp is not None:
            assert stamp_msg != builtin_interfaces.msg.Time(), "Expected 'stamp' to either be None or Unix epoch time greater zero."
            assert frame is not None, "Expected 'stamp' to not be None when stamp is provided."

    # create pose
    if isinstance(point, Point):
        point_msg = point
    else:
        point_msg = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
    if isinstance(quaternion, Quaternion):
        quaternion_msg = quaternion
    else:
        quaternion_msg = Quaternion(x=float(quaternion[0]), y=float(quaternion[1]), z=float(quaternion[2]), w=float(quaternion[3]))
    pose_msg = Pose(position=point_msg, orientation=quaternion_msg)
    if frame is None and stamp is None:
        return pose_msg
    else:
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose_msg
        if frame is not None:
            pose_stamped.header.frame_id = frame
        if stamp is not None:
            pose_stamped.header.stamp = stamp_msg
        return pose_stamped

def create_transform(vector, quaternion, frame=None, stamp=None, strict=True):
    """
    Create a Transform or TransformStamped.

    Args:
        vector (Vector3, numpy.ndarray, list, tuple): Position ([x, y, z] when using array type).
        quaternion (Quaternion, numpy.ndarray, list, tuple): Quaternion ([qx, qy, qz, qw] when using array type).
        frame (str | None, optional): If provided, returns TransformStamped with this frame. Defaults to None.
        stamp (see convert_stamp() | None, optional): If provided, returns TransformStamped with this time stamp. Defaults to None.
        strict (bool, optional): If True, asserts that frame is not empty and stamp is not zero, if at least one of the two is not None. Defaults to True.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        Transform | TransformStamped: Created transform.
    """
    # parse arguments
    assert_type_value(obj=vector, type_or_value=[Vector3, np.ndarray, list, tuple], name="argument 'vector'")
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    if not isinstance(vector, Vector3):
        assert len(vector) == 3, f"Expected 'vector' to have '3' elements (x, y, z) but got '{len(vector)}'."
        for item in vector:
            assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'vector'")
    assert_type_value(obj=quaternion, type_or_value=[Quaternion, np.ndarray, list, tuple], name="argument 'quaternion'")
    if isinstance(quaternion, np.ndarray):
        quaternion = quaternion.tolist()
    if not isinstance(quaternion, Quaternion):
        assert len(quaternion) == 4, f"Expected 'quaternion' to have '3' elements (qx, qy, qz, qw) but got '{len(quaternion)}'."
        for item in quaternion:
            assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'quaternion'")
    assert_type_value(obj=frame, type_or_value=[str, None], name="argument 'frame'")
    assert_type_value(obj=strict, type_or_value=bool, name="argument 'strict'")
    if stamp is not None:
        stamp_msg = convert_stamp(stamp=stamp, target_format="msg")
    if strict:
        if frame is not None:
            assert frame != "", "Expected 'frame' to either be None or a non-empty string."
            assert stamp is not None, "Expected 'stamp' to not be None when frame is provided."
        if stamp is not None:
            assert stamp_msg != builtin_interfaces.msg.Time(), "Expected 'stamp' to either be None or Unix epoch time greater zero."
            assert frame is not None, "Expected 'stamp' to not be None when stamp is provided."

    # create transform
    if isinstance(vector, Vector3):
        vector_msg = vector
    else:
        vector_msg = Vector3(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))
    if isinstance(quaternion, Quaternion):
        quaternion_msg = quaternion
    else:
        quaternion_msg = Quaternion(x=float(quaternion[0]), y=float(quaternion[1]), z=float(quaternion[2]), w=float(quaternion[3]))
    transform_msg = Transform(translation=vector_msg, rotation=quaternion_msg)
    if frame is None and stamp is None:
        return transform_msg
    else:
        transform_stamped = TransformStamped()
        transform_stamped.transform = transform_msg
        if frame is not None:
            transform_stamped.header.frame_id = frame
        if stamp is not None:
            transform_stamped.header.stamp = stamp_msg
        return transform_stamped

def create_pointcloud(points, frame=None, stamp=None, strict=True):
    """
    Create a PointCloud2 message from a set of 3D points.

    Args:
        points (numpy.ndarray, list, tuple): Array-like of shape (N, 3) or flat list of length 3*N.
        frame (str | None, optional): If provided. frame is used in header. Defaults to None.
        stamp (see convert_stamp() | None, optional): If provided, stamp is used in header. Defaults to None.
        strict (bool, optional): If True, asserts that frame is non-empty when provided and stamp is non-zero when provided. Defaults to True.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        PointCloud2: Created pointcloud.
    """
    # parse arguments
    assert_type_value(obj=points, type_or_value=[np.ndarray, list, tuple], name="argument 'points'")
    assert_type_value(obj=frame, type_or_value=[str, None], name="argument 'frame'")
    assert_type_value(obj=strict, type_or_value=bool, name="argument 'strict'")
    if stamp is not None:
        stamp_msg = convert_stamp(stamp=stamp, target_format="msg")
    if strict:
        if frame is not None:
            assert frame != "", "Expected 'frame' to either be None or a non-empty string."
            assert stamp is not None, "Expected 'stamp' to not be None when frame is provided."
        if stamp is not None:
            assert stamp_msg != builtin_interfaces.msg.Time(), "Expected 'stamp' to either be None or Unix epoch time greater zero."
            assert frame is not None, "Expected 'stamp' to not be None when stamp is provided."

    # prepare header
    header = Header()
    if frame is not None:
        header.frame_id = frame
    if stamp is not None:
        header.stamp = stamp_msg

    # convert to NumPy array of shape (N,3)
    pts = np.array(points, dtype=np.float32)
    if pts.ndim == 1:
        assert pts.size % 3 == 0, f"Expected 'points' with one dimension to be a flat array of length 'multiple of 3' but got '{pts.size}'."
        pts = pts.reshape(-1, 3)
    else:
        assert pts.ndim == 2 and pts.shape[1] == 3, f"Expected 'points' of shape (N,3) but got '{pts.shape}'."

    # flatten data and pack into bytes
    flat = pts.flatten()
    buf = struct.pack(f"{flat.size}f", *flat)

    # define PointField layout for XYZ floats
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    # create pointcloud
    cloud = PointCloud2(
        header=header,
        height=1,
        width=pts.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=12,
        row_step=12 * pts.shape[0],
        data=buf
    )

    return cloud

def convert_transform(transform_stamped, target_format="default"):
    """
    Convert a TransformStamped message to various target formats.

    Args:
        transform_stamped (TransformStamped): Input transformation to convert.
        target_format (str, optional): Output format to convert to. Capitalization is ignored.
            Supported values:
            - 'default': NumPy array of 7 floats (x, y, z, qx, qy, qz, qw)
            - 'mat': 4x4 NumPy transformation matrix
            - 'euler': NumPy array of 6 floats (x, y, z, yaw, pitch, roll)
            - 'rt': tuple of (3x3 rotation matrix, translation vector)
            - 'Transform': geometry_msgs.msg.Transform
            - 'TransformStamped': geometry_msgs.msg.TransformStamped
            - 'Point': geometry_msgs.msg.Point
            - 'PointStamped': geometry_msgs.msg.PointStamped
            - 'Pose': geometry_msgs.msg.Pose
            - 'PoseStamped': geometry_msgs.msg.PoseStamped
            - 'Vector3': geometry_msgs.msg.Vector3
            - 'Vector3Stamped': geometry_msgs.msg.Vector3Stamped
            - 'Quaternion': geometry_msgs.msg.Quaternion
            - 'QuaternionStamped': geometry_msgs.msg.QuaternionStamped
            Defaults to 'default'.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        object: Converted transform in the requested format.
    """
    # parse arguments
    assert_type_value(obj=transform_stamped, type_or_value=TransformStamped, name="argument 'transform_stamped'")
    assert_type_value(obj=target_format, type_or_value=str, name="argument 'target_format'")
    target_format = target_format.lower()
    supported_formats = ["default", "mat", "euler", "transform", "transformstamped", "point", "pointstamped", "pose", "posestamped", "vector3", "vector3stamped", "quaternion", "quaternionstamped", "rt"]
    assert_type_value(obj=target_format, type_or_value=supported_formats, name="argument 'transform_stamped'")

    # convert
    if target_format in ["default", "mat", "euler", "rt"]:
        x, y, z = transform_stamped.transform.translation.x, transform_stamped.transform.translation.y, transform_stamped.transform.translation.z
        qx, qy, qz, qw = transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w
        trs_vec = np.array([x, y, z, qx, qy, qz, qw])
        if target_format == "default":
            result = trs_vec
        elif target_format == "mat":
            result = np.eye(4)
            result[:3, :3] = quaternion_to_rotation_matrix(trs_vec[3:])
            result[:3, -1] = trs_vec[:3]
        elif target_format == "euler":
            result = np.zeros(6)
            result[3:] = quaternion_to_yaw_pitch_roll(trs_vec[3:])
            result[:3] = trs_vec[:3]
        elif target_format == "rt":
            result = (quaternion_to_rotation_matrix(trs_vec[3:]), trs_vec[:3])
    else:
        if target_format == "transform":
            result = transform_stamped.transform
        elif target_format == "transformstamped":
            result = transform_stamped
        elif target_format == "point":
            result = Point()
            result.x = transform_stamped.transform.translation.x
            result.y = transform_stamped.transform.translation.y
            result.z = transform_stamped.transform.translation.z
        elif target_format == "pointstamped":
            result = PointStamped()
            result.header = transform_stamped.header
            result.point.x = transform_stamped.transform.translation.x
            result.point.y = transform_stamped.transform.translation.y
            result.point.z = transform_stamped.transform.translation.z
        elif target_format == "pose":
            result = Pose()
            result.position.x = transform_stamped.transform.translation.x
            result.position.y = transform_stamped.transform.translation.y
            result.position.z = transform_stamped.transform.translation.z
            result.orientation = transform_stamped.transform.rotation
        elif target_format == "posestamped":
            result = PoseStamped()
            result.header = transform_stamped.header
            result.pose.position.x = transform_stamped.transform.translation.x
            result.pose.position.y = transform_stamped.transform.translation.y
            result.pose.position.z = transform_stamped.transform.translation.z
            result.pose.orientation = transform_stamped.transform.rotation
        elif target_format == "vector3":
            result = transform_stamped.transform.translation
        elif target_format == "vector3stamped":
            result = Vector3Stamped()
            result.header = transform_stamped.header
            result.vector = transform_stamped.transform.translation
        elif target_format == "quaternion":
            result = transform_stamped.transform.rotation
        elif target_format == "quaternionstamped":
            result = QuaternionStamped()
            result.header = transform_stamped.header
            result.quaternion = transform_stamped.transform.rotation

    return result

def convert_point_and_vector(point_or_vector):
    """
    Convert a Point or PointStamped to a Vector3 or Vector3Stamped and vice versa.

    Args:
        point_or_vector (Point, PointStamped, Vector3, Vector3Stamped): Point or vector to be converted.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        Point | PointStamped | Vector3 | Vector3Stamped: Converted message.
    """
    assert_type_value(obj=point_or_vector, type_or_value=[Point, PointStamped, Vector3, Vector3Stamped], name="argument 'point_or_vector'")
    if isinstance(point_or_vector, Point):
        vector = Vector3()
        vector.x = point_or_vector.x
        vector.y = point_or_vector.y
        vector.z = point_or_vector.z
        return vector
    elif isinstance(point_or_vector, PointStamped):
        vector_stamped = Vector3Stamped()
        vector_stamped.header = point_or_vector.header
        vector_stamped.vector.x = point_or_vector.point.x
        vector_stamped.vector.y = point_or_vector.point.y
        vector_stamped.vector.z = point_or_vector.point.z
        return vector_stamped
    elif isinstance(point_or_vector, Vector3):
        point = Point()
        point.x = point_or_vector.x
        point.y = point_or_vector.y
        point.z = point_or_vector.z
        return point
    elif isinstance(point_or_vector, Vector3Stamped):
        point_stamped = PointStamped()
        point_stamped.header = point_or_vector.header
        point_stamped.point.x = point_or_vector.vector.x
        point_stamped.point.y = point_or_vector.vector.y
        point_stamped.point.z = point_or_vector.vector.z
        return point_stamped
    else:
        raise NotImplementedError(f"Unknown 'point_or_vector' type '{type(point_or_vector).__name__}'.")

def convert_pose_and_transform(pose_or_transform):
    """
    Convert a Pose or PoseStamped to a Transform or TransformStamped and vice versa.

    Args:
        pose_or_transform (Pose, PoseStamped, Transform, TransformStamped): Pose or transform to be converted.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        Pose | PoseStamped | Transform | TransformStamped: Converted message.
    """
    assert_type_value(obj=pose_or_transform, type_or_value=[Pose, PoseStamped, Transform, TransformStamped], name="argument 'pose_or_transform'")
    if isinstance(pose_or_transform, Pose):
        transform = Transform()
        transform.translation = convert_point_and_vector(pose_or_transform.position)
        transform.rotation = pose_or_transform.orientation
        return transform
    elif isinstance(pose_or_transform, PoseStamped):
        transform_stamped = TransformStamped()
        transform_stamped.header = transform_stamped.header
        transform_stamped.transform.translation = convert_point_and_vector(pose_or_transform.pose.position)
        transform_stamped.transform.rotation = pose_or_transform.pose.orientation
        return transform_stamped
    elif isinstance(pose_or_transform, Transform):
        pose = Pose()
        pose.position = convert_point_and_vector(pose_or_transform.translation)
        pose.orientation = pose_or_transform.rotation
        return pose
    elif isinstance(pose_or_transform, TransformStamped):
        pose_stamped = PoseStamped()
        pose_stamped.header = pose_stamped.header
        pose_stamped.pose.position = convert_point_and_vector(pose_or_transform.transform.translation)
        pose_stamped.pose.orientation = pose_or_transform.transform.rotation
        return pose_stamped
    else:
        raise NotImplementedError(f"Unknown 'pose_or_transform' type '{type(pose_or_transform).__name__}'.")

def laserscan_to_pointcloud(laserscan):
    """
    Convert a LaserScan to PointCload2.

    Args:
        laserscan (LaserScan): Laser scan to be converted.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        PointCloud2: Converted message.
    """
    # parse arguments
    assert_type_value(obj=laserscan, type_or_value=LaserScan, name="argument 'laserscan'")

    # collect points
    points = []
    angle = laserscan.angle_min
    for i in range(len(laserscan.ranges)):
        if np.isfinite(laserscan.ranges[i]):
            px = np.cos(angle) * laserscan.ranges[i]
            py = np.sin(angle) * laserscan.ranges[i]
            points.append([px, py, 0.0])
        angle += laserscan.angle_increment

    # create PointCloud2
    cloud = pc2.create_cloud_xyz32(laserscan.header, points)

    return cloud

def message_to_numpy(msg):
    """
    Convert various ROS2 geometry and sensor messages to NumPy.

    Args:
        msg(see below): A ROS2 message to be converted to NumPy.
        Supported types:
        - Point, PointStamped -> (3,)
        - Vector3, Vector3Stamped -> (3,)
        - Pose, PoseStamped -> (position(3,), orientation(4,))
        - Transform, TransformStamped -> (translation(3,), rotation(4,))
        - PointCloud2 -> (N,3) or (N,6) if 'rgb' present

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
      numpy.ndarray | tuple[numpy.ndarray]: Converted message.
    """
    assert_type_value(obj=msg, type_or_value=[Point, PointStamped, Vector3, Vector3Stamped, Pose, PoseStamped, Transform, TransformStamped, PointCloud2], name="argument 'msg'")

    # Point
    if isinstance(msg, Point) or isinstance(msg, PointStamped):
        point = msg.point if isinstance(msg, PointStamped) else msg
        return np.array([point.x, point.y, point.z], dtype=np.float32)

    # Pose
    if isinstance(msg, Pose) or isinstance(msg, PoseStamped):
        pose = msg.pose if isinstance(msg, PoseStamped) else msg
        pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        ori = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=np.float32)
        return pos, ori

    # Transform
    if isinstance(msg, Transform) or isinstance(msg, TransformStamped):
        tm = msg.transform if isinstance(msg, TransformStamped) else msg
        trans = np.array([tm.translation.x, tm.translation.y, tm.translation.z], dtype=np.float32)
        rot = np.array([tm.rotation.x, tm.rotation.y, tm.rotation.z, tm.rotation.w], dtype=np.float32)
        return trans, rot

    # Vector3
    if isinstance(msg, Vector3) or isinstance(msg, Vector3Stamped):
        vec = msg.vector if isinstance(msg, Vector3Stamped) else msg
        return np.array([vec.x, vec.y, vec.z], dtype=np.float32)

    # PointCloud2
    if isinstance(msg, PointCloud2):
        def convert_rgbUint32_to_tuple(rgb_uint32):
            return (rgb_uint32 & 0x00ff0000) >> 16, \
                   (rgb_uint32 & 0x0000ff00) >> 8, \
                   (rgb_uint32 & 0x000000ff)

        def convert_rgbFloat_to_tuple(rgb_float):
            return int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)

        field_names = [field.name for field in msg.fields]
        if "rgb" in field_names:
            idx_rgb = 3
            msg.fields[idx_rgb].datatype = 6  # hacky fix to prevent interpreting colors as NaN
        points = pc2.read_points(msg, skip_nans=True, field_names=field_names)
        cloud_data = list(points)
        if len(cloud_data) == 0:
            return np.array([])

        if "rgb" in field_names:
            xyz = [(x, y, z) for x, y, z, rgb in cloud_data]
            if isinstance(cloud_data[0][idx_rgb], float):
                rgb = [convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data]
            else:
                rgb = [convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data]
            return np.concatenate([np.array(xyz), np.array(rgb) / 255.0], axis=1)
        else:
            xyz = [(d[0], d[1], d[2]) for d in cloud_data]
            return np.array(xyz)

    raise NotImplementedError(f"Unsupported message type '{type(msg).__name__}'.")

# miscellaneous

def do_transform(point_or_pose_or_cloud, transform_stamped):
    """
    Apply a transformation to various datatypes.

    Args:
        point_or_pose_or_cloud (Point, PointStamped, Pose, PoseStamped, PointCloud2): Input geometry in a known frame. Non-stamped inputs are assumed to be in 'child_frame_id' of provided `transform_stamped`.
        transform_stamped (TransformStamped): will be applied to 'point_or_pose_or_cloud'

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        transformed_point_or_pose_or_cloud (PointStamped, PoseStamped, PoseStamped): Transformed geometry.
    """
    # parse arguments
    assert_type_value(obj=point_or_pose_or_cloud, type_or_value=[Point, PointStamped, Pose, PoseStamped, PointCloud2], name="argument 'point_or_pose_or_cloud'")
    assert_type_value(obj=transform_stamped, type_or_value=TransformStamped, name="argument 'transform_stamped'")
    if isinstance(point_or_pose_or_cloud, Point):
        point_stamped = PointStamped()
        point_stamped.header.frame_id = transform_stamped.child_frame_id
        point_stamped.point = point_or_pose_or_cloud
        point_or_pose_or_cloud = point_stamped
    elif isinstance(point_or_pose_or_cloud, Pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = transform_stamped.child_frame_id
        pose_stamped.pose = point_or_pose_or_cloud
        point_or_pose_or_cloud = pose_stamped
    else:
        assert point_or_pose_or_cloud.header.frame_id == transform_stamped.child_frame_id, f"Expected child_frame_id '{transform_stamped.child_frame_id}' of the transform to match the frame_id '{point_or_pose_or_cloud.header.frame_id}' of the input geometry object."

    # transform
    if isinstance(point_or_pose_or_cloud, PointStamped):
        point_or_pose_or_cloud = do_transform_point(point_or_pose_or_cloud, transform_stamped)
    elif isinstance(point_or_pose_or_cloud, PoseStamped):
        point_or_pose_or_cloud = do_transform_pose_stamped(point_or_pose_or_cloud, transform_stamped)
    elif isinstance(point_or_pose_or_cloud, PointCloud2):
        point_or_pose_or_cloud = do_transform_cloud(point_or_pose_or_cloud, transform_stamped)

    return point_or_pose_or_cloud

def rotate_pose(pose, delta, axis="z", degrees=False):
    """
    Rotate a Pose or PoseStamped around one of its local axes.

    Args:
        pose (Pose | PoseStamped): Pose to be rotated.
        delta (int | float): Rotation angle to add.
        axis (str, optional): Axis to rotate around: 'x', 'y', or 'z'. Defaults to 'z'.
        degrees (bool, optional): If True, interpret `delta` as degrees instead of radians. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.
        ImportError: If `scipy` is not available.

    Returns:
        Pose | PoseStamped: Pose with updated orientation.
    """
    # parse arguments
    assert_type_value(obj=pose, type_or_value=[Pose, PoseStamped], name="argument 'pose'")
    assert_type_value(obj=delta, type_or_value=[int, float], name="argument 'delta'")
    assert_type_value(obj=axis, type_or_value=['x', 'y', 'z'], name="argument 'axis'")
    assert_type_value(obj=degrees, type_or_value=bool, name="argument 'degrees'")

    # extract pose
    if isinstance(pose, PoseStamped):
        original_pose = pose.pose
    else:
        original_pose = pose

    # extract quaternion
    q = [original_pose.orientation.x, original_pose.orientation.y,
         original_pose.orientation.z, original_pose.orientation.w]

    # compute updated orientation
    from scipy.spatial.transform import Rotation
    current_euler = Rotation.from_quat(q).as_euler('xyz', degrees=degrees)
    if axis == 'x':
        new_euler = (current_euler[0] + delta, current_euler[1], current_euler[2])
    elif axis == 'y':
        new_euler = (current_euler[0], current_euler[1] + delta, current_euler[2])
    else:  # 'z'
        new_euler = (current_euler[0], current_euler[1], current_euler[2] + delta)

    new_quat = Rotation.from_euler('xyz', new_euler, degrees=degrees).as_quat()

    # update pose
    updated_pose = Pose()
    updated_pose.position.x = original_pose.position.x
    updated_pose.position.y = original_pose.position.y
    updated_pose.position.z = original_pose.position.z
    updated_pose.orientation.x = new_quat[0]
    updated_pose.orientation.y = new_quat[1]
    updated_pose.orientation.z = new_quat[2]
    updated_pose.orientation.w = new_quat[3]

    if isinstance(pose, PoseStamped):
        pose_out = PoseStamped()
        pose_out.header = pose.header
        pose_out.pose = updated_pose
        return pose_out
    else:
        return updated_pose

def rotation_from_vectors(vec1, vec2, rotation_format="quat"):
    """
    Compute the rotation that aligns vec1 to vec2.

    Args:
        vec1 (numpy.ndarray | list | tuple): Non-zero source vector [x, y, z].
        vec2 (numpy.ndarray | list | tuple): Non-zero destination vector [x, y, z].
        rotation_format (str, optional): Use 'quat' to return quaternion [qx, qy, qz, qw] or 'mat' to return a 3×3 rotation matrix. Defaults to 'quat'.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        numpy.ndarray: Unit quaternion or rotation matrix aligning vec1 to vec2.
    """
    # parse arguments
    assert_type_value(obj=vec1, type_or_value=[np.ndarray, list, tuple], name="argument 'vec1'")
    assert_type_value(obj=vec2, type_or_value=[np.ndarray, list, tuple], name="argument 'vec2'")
    a = np.asarray(vec1, dtype=float)
    b = np.asarray(vec2, dtype=float)
    assert a.shape == (3,), f"Expected 'vec1' to have shape '(3,)' but got '{a.shape}'."
    assert b.shape == (3,), f"Expected 'vec2' to have shape '(3,)' but got '{b.shape}'."
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    assert norm_a > 0, "Expected 'vec1' to have non-zero length."
    assert norm_b > 0, "Expected 'vec1' to have non-zero length."
    assert_type_value(obj=rotation_format, type_or_value=["quat", "mat"], name="argument 'rotation_format'")

    # normalize
    a /= norm_a
    b /= norm_b

    # compute rotation
    cross = np.cross(a, b)
    cross_mag = np.linalg.norm(cross)
    dot = np.dot(a, b)
    if cross_mag > 1e-8:
        if rotation_format == "quat":
            axis = cross / cross_mag
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            q = np.hstack((axis * np.sin(angle / 2), np.cos(angle / 2)))
            return q / np.linalg.norm(q)
        kmat = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - dot) / (cross_mag**2))
    if dot > 0:
        return np.array([0., 0., 0., 1.]) if rotation_format == "quat" else np.eye(3)
    axis = np.array([1, 0, 0])
    if abs(a[0]) > 0.9:
        axis = np.array([0, 1, 0])
    axis = np.cross(a, axis) / np.linalg.norm(np.cross(a, axis))
    if rotation_format == "quat":
        return np.hstack((axis, 0.0))
    return -np.eye(3)

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two numeric vectors.

    Args:
        vec1 (numpy.ndarray | list | tuple): Non-zero vector [x, y, z].
        vec2 (numpy.ndarray | list | tuple): Non-zero vector [x, y, z].

    Returns:
        float: Cosine similarity value in [-1.0, +1.0].
    """
    # parse arguments
    assert_type_value(obj=vec1, type_or_value=[np.ndarray, list, tuple], name="argument 'vec1'")
    assert_type_value(obj=vec2, type_or_value=[np.ndarray, list, tuple], name="argument 'vec2'")
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    assert norm_a > 0, "Expected 'vec1' to have non-zero length."
    assert norm_b > 0, "Expected 'vec1' to have non-zero length."

    # compute similarity
    similarity = np.dot(vec1, vec2) / (norm_a * norm_b)

    return similarity

def quaternion_distance(quat0, quat1, degrees=False):
    """
    Compute the geodesic distance between two quaternions on the unit quaternion manifold.

    Args:
        quat0 (Quaternion, numpy.ndarray, list, tuple): Quaternion with non-zero norm ([qx, qy, qz, qw] when using array type).
        quat1 (Quaternion, numpy.ndarray, list, tuple): Quaternion with non-zero norm ([qx, qy, qz, qw] when using array type).
        degrees (bool, optional): If True, return angle in degrees instead of radians. Defaults to False.

    Raises:
        AssertionError: If either quaternion has invalid type, incorrect shape,
            zero norm, or contains non-finite values.

    Returns:
        float: The angular distance between the normalized quaternions in radians or degrees.
    """
    # parse arguments
    assert_type_value(obj=quat0, type_or_value=[Quaternion, np.ndarray, list, tuple], name="argument 'quat0'")
    if isinstance(quat0, Quaternion):
        quat0 = np.array([quat0.x, quat0.y, quat0.z, quat0.w], dtype=float)
    elif isinstance(quat0, np.ndarray):
        assert quat0.shape == (4,), f"Expected 'quat0' to have shape '(4,)' (qx, qy, qz, qw) but got '{quat0.shape}'."
    else:
        assert len(quat0) == 4, f"Expected 'quat0' to have '4' elements (qx, qy, qz, qw) but got '{len(quat0)}'."
    quat0 = np.asarray(quat0, dtype=float)
    quat0_norm = np.linalg.norm(quat0)
    assert quat0_norm > 0, "Expected 'quat0' to have non-zero norm but got zero."
    assert not np.any(np.invert(np.isfinite(quat0))), f"Expected 'quat0' to be finite in all dimensions but got {np.isfinite(quat0)}."
    assert_type_value(obj=quat1, type_or_value=[Quaternion, np.ndarray, list, tuple], name="argument 'quat1'")
    if isinstance(quat1, Quaternion):
        quat1 = np.array([quat1.x, quat1.y, quat1.z, quat1.w], dtype=float)
    elif isinstance(quat1, np.ndarray):
        assert quat1.shape == (4,), f"Expected 'quat1' to have shape '(4,)' (qx, qy, qz, qw) but got '{quat1.shape}'."
    else:
        assert len(quat1) == 4, f"Expected 'quat1' to have '4' elements (qx, qy, qz, qw) but got '{len(quat1)}'."
    quat1 = np.asarray(quat1, dtype=float)
    quat1_norm = np.linalg.norm(quat1)
    assert quat1_norm > 0, "Expected 'quat1' to have non-zero norm but got zero."
    assert not np.any(np.invert(np.isfinite(quat1))), f"Expected 'quat1' to be finite in all dimensions but got {np.isfinite(quat1)}."
    assert_type_value(obj=degrees, type_or_value=bool, name="argument 'degrees'")

    # normlaize
    quat0 = quat0 / quat0_norm
    quat1 = quat1 / quat1_norm

    # shortcut if same
    if np.allclose(quat0, quat1):
        return 0.0

    # compute distance
    inner = 2 * np.inner(quat0, quat1)**2 - 1
    inner = np.clip(inner, 0.0, 1.0)
    theta_rad = np.arccos(inner)

    # format
    if degrees:
        theta_deg = theta_rad * 180 / np.pi
        return theta_deg
    else:
        return theta_rad

def get_circle_pixels(center, radius, n, fill=False, image_shape=None):
    """
    Retrieve pixel coordinates on or within a radius around a point in an image.

    Args:
        center (numpy.ndarray | list | tuple): The coordinates (int | float) of the center point (x, y).
        radius (int, float): The radius around the center point in pixels to define the considered area.
        n (int): The number of unique pixel coordinates sampled on or inside the radius.
        fill (bool, optional): If True, uniformly sample coordinates inside the radius instead of only on it. Defaults to False.
        image_shape (numpy.ndarray | list | tuple | None, optional): The dimensions of the image (w, h) used to filter outliers.
                                                                     This is done after sampling, affecting the number of returned coordinates.
                                                                     Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        numpy.ndarray: Array of shape (2, <=n) and type int containing all sampled pixel coordinates.
    """
    # parse arguments
    assert_type_value(obj=center, type_or_value=[np.ndarray, list, tuple], name="argument 'center'")
    if isinstance(center, np.ndarray):
        center = center.tolist()
    assert len(center) == 2, f"Expected 'center' to contain '2' values but got '{len(center)}'."
    for item in center:
        assert_type_value(obj=item, type_or_value=[int, float], name="all items in argument 'center'")
        assert item > 0, f"Expected all values in 'center' to be greater zero but got '{item}'."
    center = np.round(center).astype(int)
    assert_type_value(obj=radius, type_or_value=[int, float], name="argument 'radius'")
    radius = int(round(radius))
    assert_type_value(obj=n, type_or_value=int, name="argument 'n'")
    assert_type_value(obj=fill, type_or_value=bool, name="argument 'fill'")
    assert_type_value(obj=image_shape, type_or_value=[np.ndarray, list, tuple, None], name="argument 'image_shape'")
    if image_shape is not None:
        if isinstance(image_shape, np.ndarray):
            image_shape = image_shape.tolist()
        assert len(image_shape) == 2, f"Expected 'image_shape' to contain '2' values but got '{len(image_shape)}'."
        for item in image_shape:
            assert_type_value(obj=item, type_or_value=int, name="all items in argument 'image_shape'")
            assert item > 0, f"Expected all values in 'image_shape' to be greater zero but got '{item}'."

    # compute circle pixels
    if fill:
        xx, yy = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        distances = np.sqrt(xx**2 + yy**2)
        mask = distances <= radius
        pixels = np.column_stack((xx[mask], yy[mask]))
        sample = np.random.choice(pixels.shape[0], size=min(n, pixels.shape[0]), replace=False)
        pixels = pixels[sample]
        pixels = pixels + center
    else:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)
        pixels = center + np.column_stack((xs, ys)) * radius
        pixels = np.round(pixels).astype(int)
        pixels = np.unique(pixels, axis=0)

    # filter image boundaries
    if image_shape is not None:
        pixels = pixels[pixels[:, 0] > 0]
        pixels = pixels[pixels[:, 0] < image_shape[0]]
        pixels = pixels[pixels[:, 1] > 0]
        pixels = pixels[pixels[:, 1] < image_shape[1]]

    return pixels

def get_iou(bbox1, bbox2):
    """
    Retrieves the intersection over union of two bounding boxes.

    Args:
        bbox1 (numpy.ndarray, list, tuple): Bounding box ([x0, y0, x1, y1] of float or int in absolute pixels, where x0 < x1 and y0 < y1).
        bbox2 (numpy.ndarray, list, tuple): Bounding box ([x0, y0, x1, y1] of float or int in absolute pixels, where x0 < x1 and y0 < y1).

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        float: IoU in [0.0, 1.0].
    """
    # parse arguments
    assert_type_value(obj=bbox1, type_or_value=[np.ndarray, list, tuple], name="argument 'bbox1'")
    assert_type_value(obj=bbox2, type_or_value=[np.ndarray, list, tuple], name="argument 'bbox2'")
    if isinstance(bbox1, np.ndarray):
        bbox1 = bbox1.tolist()
    if isinstance(bbox2, np.ndarray):
        bbox2 = bbox2.tolist()
    assert len(bbox1) == 4, f"Expected 'bbox1' to contain '4' values but got '{len(bbox1)}'."
    assert len(bbox2) == 4, f"Expected 'bbox2' to contain '4' values but got '{len(bbox2)}'."
    for item in bbox1:
        assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'bbox1'")
    for item in bbox2:
        assert_type_value(obj=item, type_or_value=[float, int], name="all items in argument 'bbox2'")
    assert bbox1[0] < bbox1[2], "Expected 'bbox1' to be bounding box (x0, y0, x1, y1) where x0 < x1."
    assert bbox1[0] < bbox1[2], "Expected 'bbox1' to be bounding box (x0, y0, x1, y1) where x0 < x1."
    assert bbox2[1] < bbox2[3], "Expected 'bbox2' to be bounding box (x0, y0, x1, y1) where y0 < y1."
    assert bbox2[1] < bbox2[3], "Expected 'bbox2' to be bounding box (x0, y0, x1, y1) where y0 < y1."

    # compute IoU

    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def convert_boxes(boxes, source_format="xyxy_absolute", target_format="xywh_absolute", image_size=None):
    """
    Convert bounding boxes between different formalisms.

    Args:
        boxes (list | tuple | numpy.ndarray):
            A list of bounding boxes. Each box must be a 4-element sequence.
        source_format (str, optional):
            Format of the input boxes. One of:
            - 'xyxy_absolute': [x_min, y_min, x_max, y_max] in pixel coords.
            - 'xyxy_normalized': [x_min, y_min, x_max, y_max] with values 0-1.
            - 'xywh_absolute': [x, y, w, h] in pixel coords.
            - 'xywh_normalized': [x, y, w, h] with values 0-1.
            Defaults to 'xyxy_absolute'.
        target_format (str, optional):
            Desired output format. See `source_format`. Defaults to 'xywh_absolute'.
        image_size (tuple | list | None, optional):
            Required if converting to or from a normalized format. Should be (height, width).
            Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        list: Converted bounding boxes in the target format.
    """
    # parse arguments
    assert_type_value(obj=boxes, type_or_value=[list, tuple, np.ndarray], name="argument 'boxes'")
    assert len(boxes) > 0, "Expected argument 'boxes' to not be an empty list."
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    for box in boxes:
        assert_type_value(obj=box, type_or_value=[list, tuple], name="all values in argument 'boxes'")
        assert len(box) == 4, f"Expected all values in argument 'boxes' to contain '4' values but got '{len(box)}'."
        for value in box:
            assert_type_value(obj=value, type_or_value=[int, float], name="all values in a single box in argument 'boxes'")
    valid_formats = ["xyxy_absolute", "xyxy_normalized", "xywh_absolute", "xywh_normalized"]
    assert_type_value(source_format, valid_formats, name="argument 'source_format'")
    assert_type_value(target_format, valid_formats, name="argument 'target_format'")
    assert_type_value(image_size, [tuple, list, None], name="argument 'image_size'")
    if ("normalized" in source_format and "absolute" in target_format) or ("absolute" in source_format and "normalized" in target_format):
        assert image_size is not None, "Expected argument 'image_size' to be a tuple (height, width) when converting between normalized and aboslute boxes."
        assert len(image_size) == 2, f"Expected argument 'image_size' to be a 2-tuple (height, width) but got a tuple of length '{len(image_size)}'."
        assert isinstance(image_size[0], int) and isinstance(image_size[1], int), "Expected argument 'image_size' to be a 2-tuple (height, width) of integers."
        h, w = image_size

    # return early if formats match
    if source_format == target_format:
        return boxes

    # flatten input and store nested structure
    def flatten(x):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 4 and all(isinstance(e, (int, float, np.integer, np.floating)) for e in x):
            return [list(x)], None
        elif isinstance(x, (list, tuple, np.ndarray)):
            flat = []
            shape = []
            for item in x:
                f, s = flatten(item)
                flat.extend(f)
                shape.append(s)
            return flat, shape
        else:
            raise TypeError("argument 'boxes' contains invalid elements")

    flat_boxes, structure = flatten(boxes)
    arr = np.asarray(flat_boxes, dtype=np.float64)

    # convert to xyxy_absolute
    if source_format == "xyxy_absolute":
        tmp = arr
    elif source_format == "xyxy_normalized":
        tmp = arr * np.array([w - 1, h - 1, w - 1, h - 1], dtype=np.float64)
    elif source_format == "xywh_absolute":
        x, y, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        tmp = np.stack([x, y, x + bw, y + bh], axis=1)
    elif source_format == "xywh_normalized":
        x = arr[:, 0] * (w - 1)
        y = arr[:, 1] * (h - 1)
        bw = arr[:, 2] * (w - 1)
        bh = arr[:, 3] * (h - 1)
        tmp = np.stack([x, y, x + bw, y + bh], axis=1)
    else:
        raise NotImplementedError(f"Unknown source format '{source_format}'.")

    # convert from xyxy_absolute to target_format
    if target_format == "xyxy_absolute":
        result = np.round(tmp).astype(int)
    elif target_format == "xyxy_normalized":
        result = tmp / np.array([w - 1, h - 1, w - 1, h - 1], dtype=np.float64)
        result = np.clip(result, 0.0, 1.0)
    elif target_format == "xywh_absolute":
        x1, y1, x2, y2 = tmp[:, 0], tmp[:, 1], tmp[:, 2], tmp[:, 3]
        result = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        result = np.round(result).astype(int)
    elif target_format == "xywh_normalized":
        x1, y1, x2, y2 = tmp[:, 0], tmp[:, 1], tmp[:, 2], tmp[:, 3]
        result = np.stack([
            x1 / (w - 1),
            y1 / (h - 1),
            (x2 - x1) / (w - 1),
            (y2 - y1) / (h - 1)
        ], axis=1)
        result = np.clip(result, 0.0, 1.0)
    else:
        raise NotImplementedError(f"Unknown target format '{target_format}'.")

    # restore original structure
    flat_result = result.tolist()
    if structure is None:
        return flat_result[0]

    def unflatten(s):
        if s is None:
            return flat_result.pop(0)
        return [unflatten(sub) for sub in s]

    result = unflatten(structure)
    return result
