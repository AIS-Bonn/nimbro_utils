#!/usr/bin/env python3

import copy
import threading

import numpy as np

import rclpy
from message_filters import Cache, Subscriber
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, LaserScan

from nimbro_utils.node_extensions.logger import Logger
from nimbro_utils.utility.image import convert_image
from nimbro_utils.utility.misc import escape, assert_type_value, assert_keys, update_dict
from nimbro_utils.utility.geometry import message_to_numpy, laserscan_to_pointcloud

default_settings = {
    # Logger severity in [10, 20, 30, 40, 50] (int).
    'severity': 20,
    # Logger suffix (str).
    'suffix': "sensors",
    # List of sensor identifiers used as keys (list[str]).
    'names': [], # ["color", "depth"],
    # List of ROS topic names corresponding to 'names' (list[str]).
    'topics': [], # ["/gemini/color/image_raw/compressed", "/gemini/depth/image_raw/compressedDepth"],
    # List of ROS message types in ['Image', 'CompressedImage', 'PointCloud2', or 'LaserScan'] corresponding to 'topics' (list[str]).
    'types': [], # ["CompressedImage", "CompressedImage"],
    # List of formats corresponding to 'names' determining what get_data() returns:
    #   - Image and CompressedImage: ["passthrough", f"{format}_{encoding}"]
    #                                with format in ["numpy", "image", "compressed"]
    #                                and encoding in ["16UC1", "mono8", "mono16", "bgr8", "rgb8", "bgr16", "rgb16"]
    #   - PointCloud2: ["passthrough", "numpy"]
    #   - LaserScan: ["passthrough", "numpy", "pointcloud"]
    'formats': [], # ["numpy_bgr8", "numpy_16UC1"],
    # Number of messages to buffer per topic (int or list[int]).
    'cache': 10
}

class SensorInterface:
    """
    Manages multiple sensor subscriptions with configurable buffering and synchronization.

    Subscribes to a configurable list of sensor topics, each with an associated message type
    and name identifier. Buffers messages per-topic and allows retrieving synchronized data
    in raw or converted (e.g., OpenCV, NumPy) format.
    """

    def __init__(self, node, settings=None):
        """
        Initialize the SensorInterface.

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

        # subscribers

        assert_type_value(obj=settings['names'], type_or_value=[list, tuple], name="setting 'names'", logger=self._logger)
        for item in settings['names']:
            assert_type_value(obj=item, type_or_value=str, name="item in setting 'names'", logger=self._logger)
        if not len(settings['names']) == len(set(settings['names'])):
            message = f"Expected all values '{settings['names']}' in setting 'names' to to be unique."
            self._logger.error(message)
            assert len(settings['names']) == len(set(settings['names'])), message

        assert_type_value(obj=settings['topics'], type_or_value=[list, tuple], name="setting 'topics'", logger=self._logger)
        for item in settings['topics']:
            assert_type_value(obj=item, type_or_value=str, name="item in setting 'topics'", logger=self._logger)
        if not len(settings['topics']) == len(set(settings['topics'])):
            message = f"Expected all values '{settings['topics']}' in setting 'topics' to to be unique."
            self._logger.error(message)
            assert len(settings['topics']) == len(set(settings['topics'])), message

        assert_type_value(obj=settings['types'], type_or_value=[list, tuple], name="setting 'types'", logger=self._logger)
        for item in settings['types']:
            assert_type_value(obj=item, type_or_value=["Image", "CompressedImage", "PointCloud2", "LaserScan"], name="item in setting 'types'", logger=self._logger)

        assert_type_value(obj=settings['formats'], type_or_value=[list, tuple], name="setting 'formats'", logger=self._logger)

        lengths = [len(settings['names']), len(settings['topics']), len(settings['types']), len(settings['formats'])]
        if not len(set(lengths)) == 1:
            message = f"Expected all settings 'names', 'topics', 'types', and 'formats' to be lists of the same length but they are {lengths}."
            self._logger.error(message)
            assert len(set(lengths)) == 1, message
        # if not lengths[0] > 0:
        #     message = "Expected all settings 'names', 'topics', 'types', and 'formats' to be non-empty lists."
        #     self._logger.error(message)
        #     assert lengths[0] > 0, message

        image_formats = []
        for f in ["numpy", "image", "compressed"]:
            for e in ["16UC1", "mono8", "mono16", "bgr8", "rgb8", "bgr16", "rgb16"]:
                image_formats.append(f"{f}_{e}")
        for i, item in enumerate(settings['formats']):
            _type = settings['types'][i]
            if _type == "Image":
                assert_type_value(obj=item, type_or_value=["passthrough"] + image_formats, name=f"item in setting 'formats' for type '{_type}'", logger=self._logger)
            elif _type == "CompressedImage":
                assert_type_value(obj=item, type_or_value=["passthrough"] + image_formats, name=f"item in setting 'formats' for type '{_type}'", logger=self._logger)
            elif _type == "PointCloud2":
                assert_type_value(obj=item, type_or_value=["passthrough", "numpy"], name=f"item in setting 'formats' for type '{_type}'", logger=self._logger)
            elif _type == "LaserScan":
                assert_type_value(obj=item, type_or_value=["passthrough", "numpy", "pointcloud"], name=f"item in setting 'formats' for type '{_type}'", logger=self._logger)

        assert_type_value(obj=settings['cache'], type_or_value=[int, list, tuple], name="setting 'cache'", logger=self._logger)
        if isinstance(settings['cache'], (list, tuple)):
            if not len(settings['cache']) == lengths[0]:
                message = f"Expected setting 'cache' to be a list of length '{lengths[0]}' but it is of length '{len(settings['cache'])}'."
                self._logger.error(message)
                assert len(settings['cache']) == len(set(settings['cache'])), message
            cache_per_topic = {}
            for i, item in enumerate(settings['cache']):
                assert_type_value(obj=item, type_or_value=int, name="item in setting 'cache'", logger=self._logger)
                if not item > 0:
                    message = f"Expected all values '{settings['cache']}' in setting 'cache' to be integers greater zero but it contains the value '{item}'."
                    self._logger.error(message)
                    assert item > 0, message
                elif cache_per_topic.get(settings['topics'][i], item) != item:
                    message = f"Expected value '{settings['topics'][i]}' in setting 'cache' for topic '{settings['topics'][i]}' to match other value '{item}' in setting 'cache' for the same topic."
                    self._logger.error(message)
                    assert cache_per_topic.get(settings['topics'][i], item) == item, message
                cache_per_topic[settings['topics'][i]] = item
        elif not settings['cache'] > 0:
            message = f"Expected value of setting '{settings['cache']}' to be an integer greater zero but it is '{settings['cache']}'."
            self._logger.error(message)
            assert settings['cache'] > 0, message

        self._settings = settings
        self.reset_subscribers(force=False)

    # subscribers

    def get_status(self, log=True):
        """
        Collect a status summary for each sensor.

        Args:
            log (bool, optional): If True, logs the status summary in a human readable form.

        Raises:
            AssertionError: If input arguments are invalid.

        Returns:
            list: Status information per sensor.
        """
        # parse arguments
        assert_type_value(obj=log, type_or_value=bool, name="argument 'log'", logger=self._logger)

        # collect names and formats per sensors
        names_and_formats = [[] for _ in range(len(self._unique_topics_and_types_and_caches))]
        for name in self._names:
            _format = self._name_to_format[name]
            names_and_formats[self._name_to_sub_and_cache_idx[name]].append((name, _format))

        # generate status
        status = []
        if log:
            text = f"Sensor status: {'- ' if len(self._unique_topics_and_types_and_caches) == 0 else ''}({len(self._unique_topics_and_types_and_caches)})"
        for i, (topic, _type, cache_size) in enumerate(self._unique_topics_and_types_and_caches):
            status.append({
                'topic': topic,
                'type': _type,
                'names_and_formats': names_and_formats[i],
                'cache_size': self._sensor_caches[i].cache_size,
                'cache_used': len(self._sensor_caches[i].cache_msgs),
            })
            if status[-1]['cache_used'] > 0:
                status[-1]['newest'] = (self._sensor_caches[i].getLastestTime() - self._node.get_clock().now()).nanoseconds * 0.000000001
            if status[-1]['cache_used'] > 1:
                status[-1]['span'] = (self._sensor_caches[i].getLastestTime() - self._sensor_caches[i].getOldestTime()).nanoseconds * 0.000000001
                if status[-1]['span'] != 0.0:
                    status[-1]['FPS'] = status[-1]['cache_used'] / status[-1]['span']

            if log:
                text += f"\n'{topic}' ({_type})"
                text += f"\n\tNames & Formats: {names_and_formats[i]}"
                text += f"\n\tCached: {status[-1]['cache_used']}/{status[-1]['cache_size']}"
                if status[-1]['cache_used'] == 0:
                    text += f" {escape['red']}(EMPTY){escape['end']}"
                if status[-1]['cache_used'] > 0:
                    text += f"\n\tNewest: {status[-1]['newest']:.3f}s"
                    if status[-1]['newest'] <= -1.0:
                        text += f" {escape['yellow']}(OLD){escape['end']}"
                if status[-1]['cache_used'] > 1:
                    text += f"\n\tSpan: {status[-1]['span']:.3f}s"
                    if status[-1]['span'] != 0.0:
                        text += f"\n\tFPS: {status[-1]['FPS']:.2f}Hz"

        # log
        if log:
            self._logger.info(text.strip())

        return status

    def reset_subscribers(self, force=True):
        """
        Create and initialize sensor subscribers and their corresponding message caches, based on the current settings.

        Args:
            force (bool, optional): Forces (re)creating subscribers when not required by settings. For internal use. Defaults to True.
        """
        assert_type_value(obj=force, type_or_value=bool, name="argument 'force'", logger=self._logger)
        self._lock.acquire()

        unique_topics_and_types_and_caches = []
        name_to_format = {}
        for i, (name, topic, _type) in enumerate(zip(self._settings['names'], self._settings['topics'], self._settings['types'])):
            if isinstance(self._settings['cache'], int):
                cache_size = self._settings['cache']
            else:
                cache_size = self._settings['cache'][i]
            if (topic, _type, cache_size) not in unique_topics_and_types_and_caches:
                unique_topics_and_types_and_caches.append((topic, _type, cache_size))
            name_to_format[name] = self._settings['formats'][i]

        if not hasattr(self, "_unique_topics_and_types_and_caches") or set(unique_topics_and_types_and_caches) != set(self._unique_topics_and_types_and_caches):
            self._logger.debug("Creating subscribers and caches")

            # qos settings
            cbg_subs = rclpy.callback_groups.ReentrantCallbackGroup()
            qos_profile_reliable = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=3)
            qos_profile_best_effort = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=3)

            self._sensor_subscribers = []
            self._sensor_caches = []
            for topic, _type, cache_size in unique_topics_and_types_and_caches:

                # subscribers
                if _type == "Image":
                    sub = Subscriber(self._node, Image, topic, qos_profile=qos_profile_best_effort, callback_group=cbg_subs)
                elif _type == "CompressedImage":
                    sub = Subscriber(self._node, CompressedImage, topic, qos_profile=qos_profile_best_effort, callback_group=cbg_subs)
                elif _type == "PointCloud2":
                    sub = Subscriber(self._node, PointCloud2, topic, qos_profile=qos_profile_best_effort, callback_group=cbg_subs)
                elif _type == "LaserScan":
                    sub = Subscriber(self._node, LaserScan, topic, qos_profile=qos_profile_reliable, callback_group=cbg_subs)
                else:
                    raise NotImplementedError(f"Topic type '{_type}'")
                self._sensor_subscribers.append(sub)

                # caches
                cache = Cache(self._sensor_subscribers[-1], cache_size=cache_size)
                self._sensor_caches.append(cache)

            self._logger.info(f"Subscribing and caching topics: {unique_topics_and_types_and_caches} ({len(unique_topics_and_types_and_caches)})")

        if not hasattr(self, "_unique_topics_and_types_and_caches") or unique_topics_and_types_and_caches != self._unique_topics_and_types_and_caches or self._names != self._settings['names']:
            self._logger.debug("Mapping names to subscribers/caches")
            self._name_to_sub_and_cache_idx = {}
            for i, (name, topic, _type) in enumerate(zip(self._settings['names'], self._settings['topics'], self._settings['types'])):
                if isinstance(self._settings['cache'], int):
                    cache_size = self._settings['cache']
                else:
                    cache_size = self._settings['cache'][i]
                self._name_to_sub_and_cache_idx[name] = unique_topics_and_types_and_caches.index((topic, _type, cache_size))

        self._unique_topics_and_types_and_caches = unique_topics_and_types_and_caches
        self._names = copy.deepcopy(self._settings['names'])
        self._name_to_format = name_to_format

        self._lock.release()

    # sensor data

    def get_data(self, source=None, max_time_delta=0.2, max_age=0.0, timeout=1.0):
        """
        Retrieve the most recent synchronized sensor data from specified sources.

        Args:
            source (str | list[str] | None, optional):
                One or more names defined in the settings to determine sensor and output format. Use None to retrieve first sensor. Defaults to None.
            max_time_delta (int | float | None, optional):
                Maximum allowed time difference (seconds) between synchronized messages. Use None to disable. Defaults to 0.2 seconds.
            max_age (float | None, optional):
                Maximum message age (seconds). Messages older than this are ignored. Use None to ignore age. Defaults to 0.0 seconds.
            timeout (float | None, optional):
                Time to wait (seconds) for messages before aborting. Use None wait forever. Defaults to 1.0 seconds.

        Raises:
            AssertionError: If input arguments are invalid.

        Returns:
            tuple[bool, str, Any]:
                - True if successful, False on error or timeout.
                - Human-readable message.
                - If successful, list or single converted message(s), depending on source type and format.
        """
        # get current stamp
        stamp_callback = self._node.get_clock().now()
        self._lock.acquire()

        # parse arguments
        assert_type_value(obj=source, type_or_value=self._names + [list, tuple, None], name="argument 'source'", logger=self._logger)
        if isinstance(source, (list, tuple)):
            for item in source:
                assert_type_value(obj=item, type_or_value=self._names, name="item in argument 'source' provided as list", logger=self._logger)
            source_was_list = True
        else:
            source_was_list = False
            if source is None:
                source = self._names[0]
                self._logger.debug(f"Using default source '{source}'")
            source = [source]
        assert_type_value(obj=max_time_delta, type_or_value=[int, float, None], name="argument 'max_time_delta'", logger=self._logger)
        if max_time_delta is not None:
            max_time_delta = abs(float(max_time_delta))
            dur_delta = rclpy.duration.Duration(seconds=max_time_delta)
            dur_step = rclpy.duration.Duration(seconds=max_time_delta * 0.2)
        assert_type_value(obj=timeout, type_or_value=[int, float, None], name="argument 'timeout'", logger=self._logger)
        if timeout is not None:
            timeout = abs(float(timeout))
            stamp_timeout = stamp_callback + rclpy.duration.Duration(seconds=timeout)
        assert_type_value(obj=max_age, type_or_value=[int, float, None], name="argument 'max_age'", logger=self._logger)
        if max_age is None:
            stamp_oldest = rclpy.time.Time(clock_type=self._node.get_clock().clock_type)
        else:
            max_age = abs(float(max_age))
            stamp_oldest = stamp_callback - rclpy.duration.Duration(seconds=max_age)

        # associate names and topics

        sub_idx = []
        name_to_sub_idx = {}
        for name in source:
            i = self._name_to_sub_and_cache_idx[name]
            if i not in sub_idx:
                sub_idx.append(i)
            name_to_sub_idx[name] = len(sub_idx) - 1
        num_idx = len(sub_idx)

        # collect data

        if len(source) == 1:
            self._logger.debug(f"Retrieving sensor data from source '{source[0]}'")
        else:
            self._logger.debug(f"Retrieving sensor data from sources: {source}")

        num_prev = -1
        stamp_now = stamp_callback
        while True:
            msgs = [self._sensor_caches[i].getInterval(stamp_oldest, stamp_now) for i in sub_idx]
            nums = np.asarray([len(i) if i is not None else 0 for i in msgs])
            num = np.sum(nums > 0)

            if num < num_idx:
                if num > num_prev:
                    if len(source) == 1:
                        message = f"Waiting for message on source '{source[0]}'"
                    else:
                        message = f"Waiting for message{'' if num_idx - num == 1 else 's'} on {num_idx - num} of {num_idx} sources {[source[i] for i in np.where(nums == 0)[0]]}"
                    self._logger.debug(message, skip_first=True, throttle_duration_sec=1.0)
                    num_prev = num
            else:
                if num_idx == 1:
                    msgs_win = [msgs[0][-1]]
                    break
                elif max_time_delta is None:
                    msgs_win = [msg[-1] for msg in msgs]
                    break

                # find newest set that satisfies max_time_delta
                stamp_window_end = stamp_now
                stamp_window_beginning = stamp_window_end - dur_delta
                while True:
                    msgs_win = []
                    for i in range(len(msgs)):
                        for j in range(len(msgs[i]) - 1, -1, -1):
                            stamp = rclpy.time.Time.from_msg(msgs[i][j].header.stamp)
                            if stamp > stamp_window_end:
                                continue
                            elif stamp < stamp_window_beginning:
                                msgs_win.append(None)
                                break
                            else:
                                msgs_win.append(msgs[i][j])
                                break
                    nums_win = np.asarray([0 if i is None else 1 for i in msgs_win])
                    num_win = np.sum(nums_win > 0)
                    if num_win == num_idx:
                        break
                    else:
                        stamp_window_end -= dur_step
                        stamp_window_beginning -= dur_step
                        if stamp_window_beginning < stamp_oldest:
                            num_prev = -1
                            stamp_oldest = stamp_now - dur_delta
                            self._logger.debug("Collected messages do not satisfy synchronicity constraints, collecting more data", throttle_duration_sec=1.0)
                            break

                if num_win == num_idx:
                    break

            stamp_now = self._node.get_clock().now()

            # timeout
            if timeout is not None and stamp_now > stamp_timeout:
                message = f"Timeout while retrieving sensor data from source{'' if len(source) == 1 else 's'} {source} after '{(stamp_now - stamp_callback).nanoseconds / 1e9:.3f}s'."
                self._logger.error(message[:-1])
                self._lock.release()
                return False, message, None

        # analysis
        stamps = np.asarray([rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds for msg in msgs_win])
        stamp_callback_s = stamp_callback.nanoseconds / 1e9
        duration_waited_s = (stamp_now - stamp_callback).nanoseconds / 1e9
        if len(source) > 1:
            info_min = np.min(stamps) / 1e9
            info_max = np.max(stamps) / 1e9
            info_span = info_max - info_min
            info_std = np.std(stamps) / 1e9
            info_min -= stamp_callback_s
            info_max -= stamp_callback_s
            if len(source) > 2:
                message = f"Retrieved sensor data from sources {source} with max. age '{-info_max:.3f}s' after waiting '{duration_waited_s:.3f}s' (max. delta: '{info_span:.3f}s'; std. delta: '{info_std:.3f}s')."
            else:
                message = f"Retrieved sensor data from sources {source} with max. age '{-info_max:.3f}s' after waiting '{duration_waited_s:.3f}s' (max. delta: '{info_span:.3f}s')."
        else:
            message = f"Retrieved sensor data from source '{source[0]}' with age '{-((stamps[0] / 1e9) - stamp_callback_s):.3f}s' after waiting '{duration_waited_s:.3f}s'."
        self._logger.debug(message[:-1])

        # conversion
        result = []
        for name in source:
            f = self._name_to_format[name]
            i = name_to_sub_idx[name]
            if f == "passthrough":
                result.append(msgs_win[i])
            else:
                if isinstance(msgs_win[i], (Image, CompressedImage)):
                    target_format, target_encoding = f.split("_")
                    converted = convert_image(image=msgs_win[i], target_format=target_format, target_encoding=target_encoding, logger=self._logger)
                elif isinstance(msgs_win[i], PointCloud2):
                    if f == "numpy":
                        converted = message_to_numpy(msgs_win[i])
                elif isinstance(msgs_win[i], LaserScan):
                    if f == "numpy":
                        converted = message_to_numpy(msgs_win[i])
                    elif f == "pointcloud":
                        converted = laserscan_to_pointcloud(msgs_win[i])
                result.append(converted)
                del converted

        if not source_was_list:
            result = result[0]

        self._lock.release()
        return True, message, result
