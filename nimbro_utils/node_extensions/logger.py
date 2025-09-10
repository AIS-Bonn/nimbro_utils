#!/usr/bin/env python3

import time
import copy
import inspect
import threading

import rclpy

from nimbro_utils.utility.misc import assert_type_value, assert_keys, assert_attribute, update_dict

# TODO write logs to file
# TODO add hooks
# TODO set RcutilsLogger environment variables

default_settings = {
    # Logger severity in [10, 20, 30, 40, 50] (int) or attribute name (str) of another Logger relative to parent node (supports nested attributes separated by dots).
    'severity': 20,
    # None to use name of parent node, attribute name (str) of another Logger relative to the passed node (supports nested attributes separated by dots), or empty string to disable prefix (str).
    'prefix': None,
    # Logger name behind prefix separated by a dot (str), or None to use prefix only (which cannot be empty string then).
    'name': None,
    # Log kept/updated settings as DEBUG instead of not logging them.
    'settings_debug': False
}

class Logger:
    """
    Wraps rclpy logging and provides easy configuration, including coupling severity and prefix/name to other Logger instances.
    """

    def __init__(self, node, settings=None):
        """
        Initialize the Logger.

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

        # prefix
        self._node_prefix = f"{self._node.get_namespace()}" \
                            f".{self._node.get_name()}".replace("/", "")
        if self._node_prefix[0] == ".":
            self._node_prefix = self._node_prefix[1:]

        # state
        self._lock = threading.Lock()
        self._settings = default_settings
        self._name, self._severity = None, None
        self._once_fired = set()
        self._skip_first_seen = set()
        self._last_throttle_time = {}
        self._method_severity = {
            'debug': 10,
            'info': 20,
            'warn': 30,
            'error': 40,
            'fatal': 50
        }
        settings = update_dict(old_dict=default_settings, new_dict=settings)
        self.set_settings(settings=settings, keep_existing=False)
        self.get_logger()

    def _log(self, method, *args, **kwargs):
        """
        Internal logging method applying filters and thread safety.

        Args:
            method (str): Logging method name ('debug', 'info', 'warn', 'error', 'fatal').
            *args: Arguments to pass to the logger message.
            **kwargs: Optional filters:
                once (bool): Log only the first occurrence.
                skip_first (bool): Skip the first occurrence.
                throttle_duration_sec (int | float | None): Minimum interval between logs.
                throttle_time_source_type (str | None): Time source for throttling ('RCUTILS_STEADY_TIME' or 'RCUTILS_SYSTEM_TIME').

        Raises:
            AssertionError: If input arguments are invalid.

        Returns:
            bool: True if the message was logged, False otherwise.
        """
        self._lock.acquire()

        # check severity
        severity = self.get_severity()
        if severity > self._method_severity[method]:
            self._lock.release()
            return False

        # parse filter arguments
        once = kwargs.get('once', False)
        assert_type_value(obj=once, type_or_value=bool, name="filter 'once'", logger=self.get_logger())
        skip_first = kwargs.get('skip_first', False)
        assert_type_value(obj=skip_first, type_or_value=bool, name="filter 'skip_first'", logger=self.get_logger())
        throttle_duration_sec = kwargs.get('throttle_duration_sec', None)
        assert_type_value(obj=throttle_duration_sec,
                          type_or_value=[int, float, None],
                          name="filter 'throttle_duration_sec'",
                          logger=self.get_logger())
        throttle_time_source_type = kwargs.get('throttle_time_source_type', None)
        assert_type_value(obj=throttle_time_source_type,
                          type_or_value=["RCUTILS_STEADY_TIME", "RCUTILS_SYSTEM_TIME", None],
                          name="filter 'throttle_time_source_type'",
                          logger=self.get_logger())

        # identify call-site key
        frame = inspect.currentframe().f_back
        key = (frame.f_code.co_name,
               frame.f_code.co_filename,
               frame.f_lineno)

        # apply filter "once"
        if once:
            if key in self._once_fired:
                self._lock.release()
                return False
            self._once_fired.add(key)
        # apply filter "skip_first"
        if skip_first:
            if key not in self._skip_first_seen:
                self._skip_first_seen.add(key)
                self._lock.release()
                return False
        # apply filter "throttle"
        if throttle_duration_sec is not None:
            if throttle_time_source_type is None or throttle_time_source_type == 'RCUTILS_STEADY_TIME':
                now = time.perf_counter()
            else:
                now = time.time()
            last = self._last_throttle_time.get((key, throttle_time_source_type))
            if last is not None and (now - last) < throttle_duration_sec:
                self._lock.release()
                return False
            self._last_throttle_time[(key, throttle_time_source_type)] = now

        # strip filter kwargs
        for k in ['once', 'skip_first', 'throttle_duration_sec', 'throttle_time_source_type']:
            kwargs.pop(k, None)

        # delegate to real logger
        logger = self.get_logger()
        logger.contexts.clear()
        logged = getattr(logger, method)(*args, **kwargs)
        self._lock.release()
        return logged

    # settings

    def get_settings(self):
        """
        Retrieve the current settings of the Logger.

        Returns:
            dict: A deep copy of the current settings.
        """
        return copy.deepcopy(self._settings)

    def set_settings(self, settings, keep_existing=True):
        """
        Update settings of the Logger.

        Args:
            settings (dict): New settings to apply.
            keep_existing (bool, optional): If True, merge with existing settings. Otherwise, replace current settings entirely. Defaults to True.

        Raises:
            AssertionError: If input arguments or provided settings are invalid.
        """
        assert_type_value(
            obj=settings,
            type_or_value=[dict, None],
            name="argument 'settings'",
            logger=self.get_logger()
        )
        assert_type_value(
            obj=keep_existing,
            type_or_value=bool,
            name="argument 'keep_existing'",
            logger=self.get_logger()
        )
        settings = update_dict(
            old_dict=self._settings if keep_existing else {},
            new_dict=settings,
            key_name="logger setting",
            logger=self.get_logger() if self.get_settings()['settings_debug'] else None,
            info=False,
            debug=self.get_settings()['settings_debug']
        )
        assert_keys(
            obj=settings,
            keys=default_settings.keys(),
            mode="match",
            name="settings",
            logger=self.get_logger()
        )

        # severity resolution and validation
        assert_type_value(
            obj=settings['severity'],
            type_or_value=[str, 10, 20, 30, 40, 50],
            name="setting 'severity'",
            logger=self.get_logger()
        )
        if isinstance(settings['severity'], str):
            # resolve linked logger severity
            logger = self
            severity = settings['severity']
            while True:
                logger = assert_attribute(
                    obj=logger._node,
                    attribute=severity,
                    exists=True,
                    name="parent node",
                    logger=self.get_logger()
                )
                assert_type_value(
                    obj=logger,
                    type_or_value=Logger,
                    name=f"node attribute '{severity}' (pointed to by setting 'severity')",
                    logger=self.get_logger()
                )
                text = "The logger severity can be coupled to another logger " \
                       "(with arbitrary recursion depth), but it cannot be coupled " \
                       "to itself (at any recursion depth)."
                if logger is self:
                    self.error(text)
                assert logger is not self, text
                severity = logger.get_settings()['severity']
                if isinstance(severity, int):
                    break

        # prefix resolution and validation
        assert_type_value(
            obj=settings['prefix'],
            type_or_value=[str, None],
            name="setting 'prefix'",
            logger=self.get_logger()
        )
        if isinstance(settings['prefix'], str) and len(settings['prefix']) > 0:
            logger = self
            prefix = settings['prefix']
            while True:
                logger = assert_attribute(
                    obj=logger._node,
                    attribute=prefix,
                    exists=True,
                    name="parent node",
                    logger=self.get_logger()
                )
                assert_type_value(
                    obj=logger,
                    type_or_value=Logger,
                    name=f"node attribute '{prefix}' (pointed to by setting 'prefix')",
                    logger=self.get_logger()
                )
                text = "The logger prefix can be coupled to another logger " \
                       "(with arbitrary recursion depth), but it cannot be coupled " \
                       "to itself (at any recursion depth)."
                if logger is self:
                    self.error(text)
                assert logger is not self, text
                prefix = logger.get_settings()['prefix']
                if prefix is None or len(prefix) == 0:
                    break

        # name and filter flags validation
        assert_type_value(
            obj=settings['name'],
            type_or_value=[str, None],
            name="setting 'name'",
            logger=self.get_logger()
        )
        if not ((settings['name'] is not None and settings['name'] != "") or settings['prefix'] != ""):
            message = "Expected setting 'name' to be of type 'str' instead of 'None', when setting 'prefix' is set to an empty string."
            self.get_logger().error(message)
            assert (settings['name'] is not None and settings['name'] != "") or settings['prefix'] != "", message
        assert_type_value(
            obj=settings['settings_debug'],
            type_or_value=bool,
            name="setting 'settings_debug'",
            logger=self.get_logger()
        )
        self._settings = settings

    def get_name(self):
        """
        Resolve and return the full Logger name based on 'prefix' and 'name' settings.

        Returns:
            str: The full Logger name.
        """
        # resolve prefix
        prefix = self.get_settings()['prefix']
        if prefix is None:
            prefix = self._node_prefix
        elif len(prefix) == 0:
            prefix = None
        else:
            logger = self
            while True:
                logger = assert_attribute(
                    obj=logger._node,
                    attribute=prefix,
                    exists=True,
                    name="parent node",
                    logger=None # avoid recursion
                )
                assert_type_value(
                    obj=logger,
                    type_or_value=Logger,
                    name=f"node attribute '{prefix}' (pointed to by setting 'prefix')",
                    logger=None
                )
                text = "The logger prefix can be coupled to another logger " \
                       "(with arbitrary recursion depth), but it cannot be coupled " \
                       "to itself (at any recursion depth)."
                assert logger is not self, text
                prefix = logger.get_settings()['prefix']
                if prefix is None or len(prefix) == 0:
                    prefix = logger.get_name()
                    break

        # resolve name
        name = self.get_settings()['name']
        assert not (prefix is None and name is None and name != ""), "Expected setting 'name' to be of type 'str' instead of 'None', when setting 'prefix' is set to an empty string."

        # combine name and prefix
        if prefix is None:
            logger_name = name
        elif name is None:
            logger_name = prefix
        else:
            logger_name = f"{prefix}.{name}"

        return logger_name

    def get_severity(self):
        """
        Resolve and return the current severity level.

        Returns:
            int: The logging severity (10-50).
        """
        severity = self.get_settings()['severity']
        if not isinstance(severity, int):
            logger = self
            while True:
                logger = assert_attribute(
                    obj=logger._node,
                    attribute=severity,
                    exists=True,
                    name="parent node",
                    logger=None
                )
                assert_type_value(
                    obj=logger,
                    type_or_value=Logger,
                    name=f"node attribute '{severity}' (pointed to by setting 'severity')",
                    logger=None
                )
                text = "The logger severity can be coupled to another logger " \
                       "(with arbitrary recursion depth), but it cannot be coupled " \
                       "to itself (at any recursion depth)."
                assert logger is not self, text
                severity = logger.get_settings()['severity']
                if isinstance(severity, int):
                    break
        return severity

    def get_logger(self):
        """
        Retrieve the underlying rclpy logger based on the current settings.

        Returns:
            rclpy.impl.rcutils_logger.RcutilsLogger: The logger instance used when logging with the current settings.
        """
        logger_name = self.get_name()
        if logger_name != self._name:
            self._name = logger_name
            self._logger = rclpy.logging.get_logger(logger_name)
            self._severity = self.get_severity()
            rclpy.logging.set_logger_level(
                name=self._name,
                level=rclpy.logging.LoggingSeverity(self._severity)
            )
        elif self.get_settings()['severity'] != self._severity:
            self._severity = self.get_severity()
            rclpy.logging.set_logger_level(
                name=self._name,
                level=rclpy.logging.LoggingSeverity(self._severity)
            )
        return self._logger

    # log

    def debug(self, *args, **kwargs):
        """
        Log a debug-level message with optional filters.

        See `_log` for further details.
        """
        return self._log("debug", *args, **kwargs)

    def info(self, *args, **kwargs):
        """
        Log an info-level message with optional filters.

        See `_log` for further details.
        """
        return self._log("info", *args, **kwargs)

    def warn(self, *args, **kwargs):
        """
        Log a warning-level message with optional filters.

        See `_log` for further details.
        """
        return self._log("warn", *args, **kwargs)

    def error(self, *args, **kwargs):
        """
        Log an error-level message with optional filters.

        See `_log` for further details.
        """
        return self._log("error", *args, **kwargs)

    def fatal(self, *args, **kwargs):
        """
        Log a fatal-level message with optional filters.

        See `_log` for further details.
        """
        return self._log("fatal", *args, **kwargs)
