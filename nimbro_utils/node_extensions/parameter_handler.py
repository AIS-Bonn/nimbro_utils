#!/usr/bin/env python3

import copy
import math
import array
import queue
import types
import threading
import traceback

import rclpy
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, ParameterType, IntegerRange, FloatingPointRange

from nimbro_utils.utility.node import SelfShutdown
from nimbro_utils.node_extensions.logger import Logger
from nimbro_utils.utility.misc import assert_type_value, assert_keys, assert_attribute, update_dict

default_settings = {
    # Logger severity in [10, 20, 30, 40, 50] (int) or attribute name (str) of a Logger relative to parent node (supports nested attributes separated by dots).
    'severity': 20,
    # Mute parameter declarations by pushing their severity to 10 (bool).
    'log_init_as_debug': True,
    # Logger suffix (str).
    'suffix': "parameters",
    # Name of the ParameterContainerProxy attribute added to the parent node (str).
    'parameters_container': "parameters",
    # Name of filter callback defined by parent node with input: (name[str], value[type of parameter], is_declared[bool]) and output: (value[type of parameter | None], message[str | None])
    'filter_callback': "filter_parameter",
    # Determine if `filter_callback` is required or optional (bool).
    'require_filter_callback': False
}

class ParameterHandler:
    """
    Manages ROS2 parameters for a node, including declaration, validation, and dynamic updates.
    """

    def __init__(self, node, settings=None):
        """
        Initialize the ParameterHandler.

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
        self._node.add_on_set_parameters_callback(self._parameter_callback)

        # logger:
        self._logger = Logger(self._node, settings={
            'severity': default_settings['severity'],
            'prefix': None,
            'name': default_settings['suffix']
        })

        # state
        self._accept_declarations = True # if False, all parameter updates are rejected
        self._accept_updates = True # if False, all parameter (un)declarations are rejected
        self._descriptors = {} # descriptors of all declared parameters
        self._parameters = ParameterContainer() # internal state of all declared parameters
        self._lock = threading.Lock() # ensures state modifications are thread-safe
        self._jobs = queue.Queue()
        self._timer = self._node.create_timer(
            timer_period_sec=0.0,
            callback=self._worker,
            callback_group=rclpy.callback_groups.MutuallyExclusiveCallbackGroup(),
            autostart=False
        )

        # settings
        settings = update_dict(old_dict=default_settings, new_dict=settings)
        self.set_settings(settings=settings, keep_existing=False)

    # settings

    def get_settings(self):
        """
        Retrieve the current settings of the ParameterHandler.

        Returns:
            dict: A deep copy of the current settings.
        """
        return copy.deepcopy(self._settings)

    def set_settings(self, settings, keep_existing=True):
        """
        Update settings of the ParameterHandler.

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

        # update logger
        self._logger.set_settings({'severity': settings['severity'], 'name': settings['suffix']})

        # require_filter_callback
        assert_type_value(obj=settings['require_filter_callback'], type_or_value=bool, name="argument 'require_filter_callback'", logger=self._logger)

        # filter_callback
        assert_type_value(obj=settings['filter_callback'], type_or_value=str, name="argument 'filter_callback'", logger=self._logger)
        if len(settings['filter_callback']) == 0:
            message = "Expected argument 'filter_callback' to be a non-empty string."
            self._logger.error(message)
            assert len(settings['filter_callback']) > 0, message
        if settings['require_filter_callback']:
            assert_attribute(obj=self._node, attribute=settings['filter_callback'], exists=True, name="parent node (pointed to by setting 'filter_callback')", logger=self._logger)
            if not callable(self._node.__getattribute__(settings['filter_callback'])):
                message = f"Expected attribute '{settings['filter_callback']}' of parent node (pointed to by setting 'filter_callback') to be callable."
                self._logger.error(message)
                assert callable(self._node.__getattribute__(settings['filter_callback'])), message

        # parameters_container
        assert_type_value(obj=settings['parameters_container'], type_or_value=str, name="argument 'parameters_container'", logger=self._logger)
        if len(settings['parameters_container']) == 0:
            message = "Expected argument 'parameters_container' to be a non-empty string."
            self._logger.error(message)
            assert len(settings['parameters_container']) > 0, message
        if hasattr(self, "_settings"):
            assert_attribute(obj=self._node, attribute=self._settings['parameters_container'], exists=True, name="parent node (pointed to by setting 'parameters_container')", logger=self._logger)
            assert_type_value(obj=self._node.__getattribute__(self._settings['parameters_container']), type_or_value=ParameterContainerProxy, name=f"attribute '{self._settings['parameters_container']}' of parent node (pointed to by setting 'parameters_container')", logger=self._logger)
            if self._node.__getattribute__(self._settings['parameters_container'])._parameters is not self._parameters:
                message = f"Expected attribute '{self._settings['parameters_container']}' of parent node to be a ParameterContainerProxy pointing to internal ParameterContainer."
                self._logger.error(message)
                assert self._node.__getattribute__(self._settings['parameters_container'])._parameters is self._parameters, message
            if self._settings['parameters_container'] != settings['parameters_container']:
                assert_attribute(obj=self._node, attribute=settings['parameters_container'], exists=False, name="parent node (pointed to by setting 'parameters_container')", logger=self._logger)
                self._logger.warn(f"Renaming parameter container of parent node from '{self._settings['parameters_container']}' to '{settings['parameters_container']}'")
                setattr(self._node, settings['parameters_container'], self._node.__getattribute__(self._settings['parameters_container']))
                delattr(self._node, self._settings['parameters_container'])
        else:
            assert_attribute(obj=self._node, attribute=settings['parameters_container'], exists=False, name="parent node (pointed to by setting 'parameters_container')", logger=self._logger)
            self._logger.debug(f"Adding parameter container '{settings['parameters_container']}' to parent node")
            setattr(self._node, settings['parameters_container'], ParameterContainerProxy(self._parameters))

        self._settings = settings

    # declarations

    def declarations_activated(self):
        """
        Check if parameter declarations are currently allowed.

        Returns:
            bool: True if declarations are currently allowed.
        """
        return copy.copy(self._accept_declarations)

    def activate_declarations(self):
        """
        Enable parameter declarations.
        """
        if self._accept_declarations:
            self._logger.debug("Declaring parameters is already activated")
        else:
            self._accept_declarations = True
            self._logger.info("Declaring parameters activated")

    def deactivate_declarations(self):
        """
        Disable parameter declarations.
        """
        if self._accept_declarations:
            self._accept_declarations = False
            self._logger.info("Declaring parameters deactivated")
        else:
            self._logger.debug("Declaring parameters is already deactivated")

    def declare(self, name, dtype, default_value, *, description="", additional_constraints="", read_only=True, range_min=None, range_max=None, range_step=None):
        """
        Declare a new ROS2 parameter.

        Args:
            name (str): Parameter name that does not exist already.
            dtype (type | None): Parameter type in [None, bool, int, float, str, list[bytes], list[bool], list[int], list[float], list[str]]. Set None for dynamic typing.
            default_value: Default value for the parameter if not overwritten by launch system.
            description (str, optional): Human-readable description of the parameter. Defaults to empty string.
            additional_constraints (str, optional): Extra constraint notes of the parameter. Defaults to empty string.
            read_only (bool, optional): If True, the parameter cannot be changed after declaration. Defaults to True.
            range_min (int | float | None, optional): Minimum allowable value. Must be None if dtype is not int or float. Defaults to None.
            range_max (int | float | None, optional): Maximum allowable value. Must be None if dtype is not int or float. Defaults to None.
            range_step (int | float | None, optional): Step size for ranges. Must be None if dtype is not int or float. Defaults to None.

        Raises:
            AssertionError: If input arguments are invalid or parameter declarations are deactivated.
        """
        # parse arguments
        assert_type_value(obj=self._accept_declarations, type_or_value=True, name="attribute '_accept_declarations'", text="Adding declarations is currently deactivated.", logger=self._logger)
        assert_type_value(obj=name, type_or_value=str, name="argument 'name'", logger=self._logger)
        if len(name) == 0:
            message = "Expected argument 'name' to be a non-empty string."
            self._logger.error(message)
            assert len(name) > 0, message
        assert_keys(obj=self._descriptors, keys=[name], mode="blacklist", name="list of declared parameters", logger=self._logger)
        assert_type_value(obj=dtype, type_or_value=[None, bool, int, float, str, list[bytes], list[bool], list[int], list[float], list[str]], match_types=False, match_types_as_values=True, name="argument 'dtype'", logger=self._logger)
        assert_type_value(obj=description, type_or_value=str, name="argument 'description'", logger=self._logger)
        assert_type_value(obj=additional_constraints, type_or_value=str, name="argument 'additional_constraints'", logger=self._logger)
        assert_type_value(obj=read_only, type_or_value=bool, name="argument 'read_only'", logger=self._logger)
        assert_type_value(obj=range_min, type_or_value=[int, float, None], name="argument 'range_min'", logger=self._logger)
        assert_type_value(obj=range_max, type_or_value=[int, float, None], name="argument 'range_max'", logger=self._logger)
        assert_type_value(obj=range_step, type_or_value=[int, float, None], name="argument 'range_step'", logger=self._logger)
        range_types = [type(range_min).__name__, type(range_max).__name__, type(range_step).__name__]
        if len(set(range_types)) > 1:
            message = f"Expected arguments 'range_min', 'range_max', and 'range_step' to be of the same type, but they are {range_types}."
            self._logger.error(message)
            assert len(set(range_types)) == 1, message
        if dtype is int:
            if not (isinstance(range_min, int) or range_min is None):
                message = f"Expected arguments 'range_min', 'range_max', and 'range_step' to be of type 'int' or 'NoneType' for parameter type '{dtype.__name__}', but they are '{type(range_min).__name__}'."
                self._logger.error(message)
                assert isinstance(range_min, int) or range_min is None, message
        elif dtype is float:
            if not (isinstance(range_min, float) or range_min is None):
                message = f"Expected arguments 'range_min', 'range_max', and 'range_step' to be of type 'float' or 'NoneType' for parameter type '{dtype.__name__}', but they are '{type(range_min).__name__}'."
                self._logger.error(message)
                assert isinstance(range_min, float) or range_min is None, message
        elif range_min is not None:
            message = f"Expected arguments 'range_min', 'range_max', and 'range_step' to be 'None' for parameter type '{dtype.__name__}', but they are '{type(range_min).__name__}'."
            self._logger.error(message)
            assert range_min is None, message

        # create descriptor
        descriptor = ParameterDescriptor()
        descriptor.name = name
        descriptor.dynamic_typing = False
        if dtype is None:
            try:
                parameter = rclpy.parameter.Parameter(name=descriptor.name, value=default_value)
            except Exception:
                message = f"Value '{default_value}' is not a valid parameter value."
                self._logger.error(message)
                assert False, message
            descriptor.type = ParameterType.PARAMETER_NOT_SET
            descriptor.dynamic_typing = True
        elif dtype == bool:
            assert_type_value(obj=default_value, type_or_value=bool, name="argument 'default_value'", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_BOOL
        elif dtype == int:
            assert_type_value(obj=default_value, type_or_value=int, name="argument 'default_value'", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_INTEGER
        elif dtype == float:
            assert_type_value(obj=default_value, type_or_value=float, name="argument 'default_value'", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_DOUBLE
        elif dtype == str:
            assert_type_value(obj=default_value, type_or_value=str, name="argument 'default_value'", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_STRING
        elif dtype == list[bytes]:
            assert_type_value(obj=default_value, type_or_value=[list, tuple], name="argument 'default_value'", logger=self._logger)
            for item in default_value:
                assert_type_value(obj=item, type_or_value=bytes, name="all items in argument 'default_value' provided as list", logger=self._logger)
                if len(item) != 1:
                    message = f"Expected all items in argument 'default_value' provided as list for parameter type '{dtype}' to be of length '1' but it contains an element of length '{len(item)}'."
                    self._logger.error(message)
                    assert len(item) == 1, message
            descriptor.type = ParameterType.PARAMETER_BYTE_ARRAY
        elif dtype == list[bool]:
            assert_type_value(obj=default_value, type_or_value=[list, tuple], name="argument 'default_value'", logger=self._logger)
            for item in default_value:
                assert_type_value(obj=item, type_or_value=bool, name="all items in argument 'default_value' provided as list", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_BOOL_ARRAY
            assert len(default_value) > 0, "Parameters cannot be declared with an empty list as default value because rclpy wrongly infers them as byte array instead of respecting the descriptor type."
        elif dtype == list[int]:
            assert_type_value(obj=default_value, type_or_value=[list, tuple, array.array], name="argument 'default_value'", logger=self._logger)
            for item in default_value:
                assert_type_value(obj=item, type_or_value=int, name="all items in argument 'default_value' provided as list", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_INTEGER_ARRAY
            assert len(default_value) > 0, "Parameters cannot be declared with an empty list as default value because rclpy wrongly infers them as byte array instead of respecting the descriptor type."
        elif dtype == list[float]:
            assert_type_value(obj=default_value, type_or_value=[list, tuple, array.array], name="argument 'default_value'", logger=self._logger)
            for item in default_value:
                assert_type_value(obj=item, type_or_value=float, name="all items in argument 'default_value' provided as list", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            assert len(default_value) > 0, "Parameters cannot be declared with an empty list as default value because rclpy wrongly infers them as byte array instead of respecting the descriptor type."
        elif dtype == list[str]:
            assert_type_value(obj=default_value, type_or_value=[list, tuple], name="argument 'default_value'", logger=self._logger)
            for item in default_value:
                assert_type_value(obj=item, type_or_value=str, name="all items in argument 'default_value' provided as list", logger=self._logger)
            descriptor.type = ParameterType.PARAMETER_STRING_ARRAY
            assert len(default_value) > 0, "Parameters cannot be declared with an empty list as default value because rclpy wrongly infers them as byte array instead of respecting the descriptor type."
        else:
            raise NotImplementedError(f"dtype '{dtype}'")

        descriptor.description = description
        descriptor.additional_constraints = additional_constraints
        descriptor.read_only = read_only
        if isinstance(range_min, int):
            int_range = IntegerRange()
            int_range.from_value = range_min
            int_range.to_value = range_max
            int_range.step = range_step
            descriptor.integer_range.append(int_range)
        elif isinstance(range_min, float):
            float_range = FloatingPointRange()
            float_range.from_value = range_min
            float_range.to_value = range_max
            float_range.step = range_step
            descriptor.floating_point_range.append(float_range)

        if dtype is not None:
            try:
                parameter = rclpy.parameter.Parameter(name=name, value=default_value, type_=list(rclpy.parameter.Parameter.Type)[descriptor.type])
            except Exception:
                message = f"Value '{default_value}' is not a valid parameter value for type '{dtype}'."
                self._logger.error(message)
                assert False, message
        success, message = self._validate_value_against_descriptor(
            parameter=parameter,
            descriptor=descriptor,
            is_decalred=False
        )
        assert success, message

        # declare parameter
        self._lock.acquire()
        self._descriptors[descriptor.name] = copy.deepcopy(descriptor)
        self._logger.debug(f"Added descriptor '{len(self._descriptors) - 1}': {self._descriptors[descriptor.name]}")
        self._node.declare_parameter(self._descriptors[descriptor.name].name, default_value, self._descriptors[descriptor.name])

        self._lock.release()
        self._logger.debug(f"Declared parameter '{self._descriptors[descriptor.name].name}': {self._descriptors[descriptor.name]}")

    def declare_descriptor(self, descriptor, default_value):
        """
        Declare a new ROS2 parameter using a ParameterDescriptor.

        Args:
            descriptor (ParameterDescriptor): Descriptor of the parameter.
            default_value: Default value for the parameter if not overwritten by launch system.

        Raises:
            AssertionError: If input arguments are invalid or parameter declarations are deactivated.
        """
        # parse arguments
        assert_type_value(obj=self._accept_declarations, type_or_value=True, name="attribute '_accept_declarations'", text="Adding declarations is currently deactivated.", logger=self._logger)
        assert_type_value(obj=descriptor, type_or_value=ParameterDescriptor, name="descriptor", logger=self._logger)
        if len(descriptor.name) == 0:
            message = "Expected attribute 'name' of argument 'descriptor' to be a non-empty string."
            self._logger.error(message)
            assert len(descriptor.name) > 0, message
        assert_keys(obj=self._descriptors, keys=[descriptor.name], mode="blacklist", name="list of declared parameters", logger=self._logger)

        # validate descriptor and default_value
        try:
            if descriptor.type == 0:
                parameter = rclpy.parameter.Parameter(name=descriptor.name, value=default_value, type_=list(rclpy.parameter.Parameter.Type)[descriptor.type])
            else:
                parameter = rclpy.parameter.Parameter(name=descriptor.name, value=default_value)
        except Exception:
            message = f"Value '{default_value}' is not a valid parameter value for type '{list(rclpy.parameter.Parameter.Type)[descriptor.type]}'."
            self._logger.error(message)
            assert False, message
        success, message = self._validate_value_against_descriptor(
            parameter=parameter,
            descriptor=descriptor,
            is_decalred=False
        )
        assert success, message

        # declare parameter
        self._lock.acquire()
        self._descriptors[descriptor.name] = copy.deepcopy(descriptor)
        self._logger.debug(f"Added descriptor '{len(self._descriptors) - 1}': {self._descriptors[descriptor.name]}")
        self._node.declare_parameter(self._descriptors[descriptor.name].name, default_value, self._descriptors[descriptor.name])
        self._lock.release()
        self._logger.debug(f"Declared parameter '{self._descriptors[descriptor.name].name}': {self._descriptors[descriptor.name]}")

    def undeclare(self, name=None):
        """
        Remove one or all declared parameters from the node.

        Args:
            name (str, optional): Name of the parameter to remove. Pass None to remove declared parameters. Defaults to None.

        Raises:
            AssertionError: If input arguments are invalid or parameter declarations are deactivated.
        """
        assert_type_value(obj=name, type_or_value=[str, None], name="argument 'name'", logger=self._logger)
        if name is None:
            self._lock.acquire()
            names = list(copy.deepcopy(self._descriptors).keys())
            for name in names:
                try:
                    self._node.undeclare_parameter(name)
                except Exception:
                    self._lock.release()
                    raise
                del self._descriptors[name]
                self._parameters.remove_parameter(name)
            self._lock.release()
            self._logger.info(f"Successfully undeclared all parameters {names}.")
        else:
            assert_keys(obj=self._descriptors, keys=[name], mode="required", name="list of declared parameters", text="Cannot undeclare parameters that have not been declared.", logger=self._logger)
            self._lock.acquire()
            try:
                self._node.undeclare_parameter(name)
            except Exception:
                self._lock.release()
                raise
            del self._descriptors[name]
            self._parameters.remove_parameter(name)
            self._lock.release()
            self._logger.info(f"Successfully undeclared parameter '{name}'.")

    # updates

    def updates_activated(self):
        """
        Check if parameter updates are currently allowed.

        Returns:
            bool: True if runtime updates are currently allowed.
        """
        return copy.copy(self._accept_updates)

    def activate_updates(self):
        """
        Enable parameter updates.
        """
        if self._accept_updates:
            self._logger.debug("Updating parameters is already activated")
        else:
            self._accept_updates = True
            self._logger.info("Updating parameters activated")

    def deactivate_updates(self):
        """
        Disable parameter updates.
        """
        if self._accept_updates:
            self._accept_updates = False
            self._logger.info("Updating parameters deactivated")
        else:
            self._logger.debug("Updating parameters is already deactivated")

    def update(self, name, value, *, ignore_redundant=True):
        """
        Update the value of a declared parameter.

        Args:
            name (str): Parameter name to update.
            value: New parameter value matching the parameter type or any parameter type when using dynamic typing.
            ignore_redundant (bool, optional): If False, explicitly set parameter to it's current value. Defaults to False.

        Raises:
            AssertionError: If input arguments are invalid.

        Returns:
            tuple[bool, str]: Success flag, status message.
        """
        assert_type_value(obj=ignore_redundant, type_or_value=bool, name="argument 'ignore_redundant'", logger=self._logger)
        parameters = self._parameters.get()
        # not declared
        if name not in parameters:
            message = f"Cannot set parameter '{name}' that has not been declared."
            self._logger.error(message)
            return False, message
        # current value
        if ignore_redundant and value == parameters[name]:
            message = f"Ignoring attempt to set parameter '{name}' to its current value '{value}'."
            self._logger.debug(message)
            return True, message
        # invalid value
        try:
            if self._descriptors[name].type == 0 or self._descriptors[name].dynamic_typing:
                parameter = rclpy.parameter.Parameter(name=name, value=value)
            else:
                parameter = rclpy.parameter.Parameter(name=name, value=value, type_=list(rclpy.parameter.Parameter.Type)[self._descriptors[name].type])
        except Exception as e:
            success = False
            message = repr(e).replace('"', "")
        else:
            success, message = self._validate_value_against_descriptor(
                parameter=parameter,
                descriptor=self._descriptors[name],
                is_decalred=True
            )
        if not success:
            message = f"Rejected attempt to update parameter '{name}' to '{value}': {message}"
            self._logger.error(message)
            return False, message
        # set parameter
        self._logger.debug(f"Setting parameter '{name}' to '{value}'")
        self._lock.acquire()
        result = self._node.set_parameters([parameter])[0]
        self._logger.debug(f"set_parameter(): {result}")
        if not result.successful:
            self._lock.release()
            return False, result.reason
        parameters = self._parameters.get()
        self._lock.release()
        if parameters[name] != value:
            return True, f"Parameter '{name}' deflected to '{parameters[name]}' instead of '{value}'."
        return True, f"Parameter '{name}' set to '{parameters[name]}'."

    def update_dict(self, parameters):
        """
        Update the value of multiple declared parameter.

        Args:
            parameters (dict): Mapping of parameter names to new values.

        Raises:
            AssertionError: If input arguments are invalid.

        Returns:
            tuple[list[bool], list[str]]: Parallel lists of success flags and messages.
        """
        assert_type_value(
            obj=parameters,
            type_or_value=dict,
            name="argument 'parameters'",
            logger=self._logger
        )
        successes, messages = [], []
        for name, value in parameters.items(): # TODO make atomic
            success, message = self.update(name, value)
            successes.append(success)
            messages.append(message)
        return successes, messages

    # internals

    def _worker(self):
        while not self._jobs.empty():
            job = self._jobs.get_nowait()
            self._logger.debug(f"Executing job {job}")
            if job['type'] == "deflection":
                self.update(name=job['name'], value=job['value'], ignore_redundant=False)
        self._timer.cancel()

    def _validate_value_against_descriptor(self, parameter, descriptor, is_decalred):
        # read only
        if descriptor.read_only and is_decalred:
            return False, "Descriptor defines parameter as read only."

        # type
        if not descriptor.dynamic_typing and descriptor.type != 0 and descriptor.type != parameter.to_parameter_msg().value._type:
            PARAMETER_TYPES = ["Type.NOT_SET", "Type.BOOL", "Type.INTEGER", "Type.DOUBLE", "Type.STRING", "Type.BYTE_ARRAY", "Type.BOOL_ARRAY", "Type.INTEGER_ARRAY", "Type.DOUBLE_ARRAY", "Type.STRING_ARRAY"]
            return False, f"Value type '{parameter.type_}' does not match descriptor type '{PARAMETER_TYPES[descriptor.type]}'."

        # integer range
        if len(descriptor.integer_range) > 0:
            assert len(descriptor.integer_range) == 1, f"Expected number of integer ranges to be '1' but it is '{len(descriptor.integer_range)}'."
            assert len(descriptor.floating_point_range) == 0, f"Expected number of floating point ranges to be '0' but it is '{len(descriptor.floating_point_range)}'."
            min_value = descriptor.integer_range[0].from_value
            max_value = descriptor.integer_range[0].to_value
            if not min_value <= parameter.value <= max_value:
                return False, f"Value '{parameter.value}' is not in integer range '{min_value}' to '{max_value}'."
            if descriptor.integer_range[0].step != 0 and parameter.value not in [min_value, max_value] and (parameter.value - min_value) % descriptor.integer_range[0].step != 0:
                return False, f"Value '{parameter.value}' is not a valid step for minimum '{min_value}' and step size '{descriptor.integer_range[0].step}'."

        # floating point range
        elif len(descriptor.floating_point_range) > 0:
            assert len(descriptor.floating_point_range) == 1, f"Expected number of floating point ranges to be '1' but it is '{len(descriptor.floating_point_range)}'."
            assert len(descriptor.integer_range) == 0, f"Expected number of integer ranges to be '0' but it is '{len(descriptor.integer_range)}'."
            min_value = descriptor.floating_point_range[0].from_value
            max_value = descriptor.floating_point_range[0].to_value
            if not min_value <= parameter.value <= max_value:
                return False, f"Value '{parameter.value}' is not in floating point range '{min_value}' to '{max_value}'."
            PARAM_REL_TOL = assert_attribute(
                obj=self._node,
                attribute="PARAM_REL_TOL",
                exists=True,
                name="parent node attribute",
                logger=self._logger
            )
            assert_type_value(
                obj=PARAM_REL_TOL,
                type_or_value=float,
                name="parent node attribute 'PARAM_REL_TOL'",
                logger=self._logger
            )
            if descriptor.floating_point_range[0].step != 0.0:
                if not (math.isclose(parameter.value, min_value, rel_tol=PARAM_REL_TOL) or math.isclose(parameter.value, max_value, rel_tol=PARAM_REL_TOL)):
                    distance_int_steps = round((parameter.value - min_value) / descriptor.floating_point_range[0].step)
                    if not math.isclose(min_value + distance_int_steps * descriptor.floating_point_range[0].step, parameter.value, rel_tol=PARAM_REL_TOL):
                        return False, f"Value '{parameter.value}' is not close enough ('{PARAM_REL_TOL}') to a valid step for minimum '{min_value}' and step size '{descriptor.floating_point_range[0].step}'."

        return True, ""

    def _parameter_callback(self, parameters):
        self._logger.debug(f"Received '{len(parameters)}' parameter change{'' if len(parameters) == 1 else 's'}: {[p.name for p in parameters]}")

        # iterate parameter changes (validation only)
        success, message, deflected, deflected_value, deflected_with_message = [], [], [], [], []
        for i in range(len(parameters)):
            update_str = "set" if parameters[i].name in self._parameters else "initialized"

            deflected.append(False)
            deflected_value.append(None)
            deflected_with_message.append(False)

            # updates deactivated
            if self._accept_updates:
                success.append(True)
                message.append("")
            else:
                success.append(False)
                message.append(f"Parameter '{parameters[i].name}' not {update_str} to '{parameters[i].value}' because parameter updates are deactivated.")

            # validate descriptors
            if success[-1]:
                if parameters[i].name in self._descriptors:
                    success[-1], message[-1] = self._validate_value_against_descriptor(
                        parameter=parameters[i],
                        descriptor=self._descriptors[parameters[i].name],
                        is_decalred=parameters[i].name in self._parameters
                    )
                else:
                    success[-1] = False
                    message[-1] = f"Parameter '{parameters[i].name}' not {update_str} to '{parameters[i].value}' because it can not be associated to a known descriptor."

                # filter callback
                if success[-1]:
                    # check filter availability
                    try:
                        assert_type_value(obj=self._settings['filter_callback'], type_or_value=str, name="argument 'filter_callback'", logger=None)
                        assert_attribute(obj=self._node, attribute=self._settings['filter_callback'], exists=True, name="parent node (pointed to by setting 'filter_callback')", logger=None)
                        if not callable(self._node.__getattribute__(self._settings['filter_callback'])):
                            text = f"Expected attribute '{self._settings['filter_callback']}' of parent node (pointed to by setting 'filter_callback') to be callable."
                            self._logger.error(text)
                            assert not callable(self._node.__getattribute__(self._settings['filter_callback'])), text
                    except AssertionError as e:
                        if self._settings['require_filter_callback']:
                            success[-1] = False
                            message[-1] = f"Parameter '{parameters[i].name}' not {update_str} to '{parameters[i].value}': Callback '{self._settings['filter_callback']}' is not available: {repr(e)}"
                        else:
                            self._logger.debug(f"Ignoring filter callback '{self._settings['filter_callback']}' because it is not available and not required.")
                    else:
                        # apply filter
                        try:
                            result = self._node.__getattribute__(self._settings['filter_callback'])(copy.copy(parameters[i].name), copy.deepcopy(parameters[i].value), parameters[i].name in self._parameters)
                        except Exception as e:
                            self._node.get_logger().error(f"{repr(e)}\n{traceback.format_exc()}")
                            if self._settings['require_filter_callback']:
                                success[-1] = False
                                message[-1] = f"Parameter '{parameters[i].name}' not {update_str} to '{parameters[i].value}': Exception in '{self._settings['filter_callback']}' callback: {repr(e)}"
                            else:
                                self._logger.warn(f"Ignoring filter callback '{self._settings['filter_callback']}' because it raised an exception and is not required.")
                        else:
                            # parse filter results
                            try:
                                _value, _message = result
                                assert_type_value(obj=_value, type_or_value=[object, None], name="result at index '0'", logger=None)
                                assert_type_value(obj=_message, type_or_value=[str, None], name="result at index '1'", logger=None)
                            except Exception as e:
                                if self._settings['require_filter_callback']:
                                    success[-1] = False
                                    message[-1] = f"Parameter '{parameters[i].name}' not {update_str} to '{parameters[i].value}': Invalid result by '{self._settings['filter_callback']}' callback: {repr(e)}"
                                else:
                                    self._logger.warn(f"Ignoring filter callback '{self._settings['filter_callback']}' because it returned an invalid result and is not required.")
                            else:
                                info = {'value': _value, 'message': _message}
                                self._logger.debug(f"Filter callback returned valid result: {info}")
                                # filtered
                                if _value is None and parameters[i].to_parameter_msg().value._type != 0:
                                    success[-1] = False
                                    if _message in [None, ""]:
                                        message[-1] = f"Parameter '{parameters[i].name}' not {update_str} to '{parameters[i].value}'."
                                    else:
                                        message[-1] = f"Parameter '{parameters[i].name}' not {update_str} to '{parameters[i].value}': {_message}"
                                # passed
                                elif _value == parameters[i].value:
                                    pass
                                # deflected
                                else:
                                    deflected[-1] = True
                                    deflected_value[-1] = _value

                                    if update_str == "initialized":
                                        update_str = "initialize"

                                    if parameters[i].name in self._parameters and self._parameters[parameters[i].name] == _value:
                                        if _message in [None, ""]:
                                            message[-1] = f"Parameter '{parameters[i].name}' deflected to its current value '{_value}' after attempt to {update_str} it to '{parameters[i].value}'."
                                        else:
                                            message[-1] = f"Parameter '{parameters[i].name}' deflected to its current value '{_value}': {_message}"
                                            deflected_with_message[-1] = True
                                    else:
                                        if _message in [None, ""]:
                                            message[-1] = f"Parameter '{parameters[i].name}' deflected to '{_value}' after attempt to {update_str} it to '{parameters[i].value}'."
                                        else:
                                            message[-1] = f"Parameter '{parameters[i].name}' deflected to '{_value}': {_message}"
                                            deflected_with_message[-1] = True

        # apply all parameters atomically if all succeeded

        all_success = all(success[i] or deflected[i] for i in range(len(parameters)))
        self._logger.debug(f"All successful: {all_success}")
        trigger_deflection_jobs = False
        for i in range(len(parameters)):
            update_str = "set" if parameters[i].name in self._parameters else "initialized"

            if all_success and (success[i] or deflected[i]):
                self._logger.debug(f"Applying parameter '{parameters[i].name}'")
                assert_attribute(
                    obj=self._node,
                    attribute=self._settings['parameters_container'],
                    exists=True,
                    name="parent node (pointed to by setting 'parameters_container')",
                    logger=self._logger
                )
                if deflected[i]:
                    if parameters[i].name in self._parameters and self._parameters[parameters[i].name] == deflected_value[i]:
                        pass
                    else:
                        trigger_deflection_jobs = True
                        self._parameters.set_parameter(parameters[i].name, deflected_value[i])
                        self._jobs.put_nowait({'type': "deflection", 'name': parameters[i].name, 'value': deflected_value[i]})
                    if deflected_with_message[i]:
                        self._logger.warn(message[i])
                    else:
                        if self._settings['log_init_as_debug'] and update_str == "initialized":
                            self._logger.debug(message[i])
                        else:
                            self._logger.info(message[i])
                else:
                    if parameters[i].name in self._parameters and self._parameters[parameters[i].name] == parameters[i].value:
                        message[i] = f"Kept parameter '{parameters[i].name}' set to '{parameters[i].value}'."
                        self._logger.debug(message[i])
                    else:
                        message[i] = f"Parameter '{parameters[i].name}' {update_str} to '{parameters[i].value}'."
                        self._parameters.set_parameter(parameters[i].name, parameters[i].value)
                        if self._settings['log_init_as_debug'] and update_str == "initialized":
                            self._logger.debug(message[i])
                        else:
                            self._logger.info(message[i])
            else:
                if not success[i]:
                    self._logger.error(message[i])
                elif deflected[i]:
                    message[i] = f"Ignored setting deflected parameter '{parameters[i].name}' due to atomic rollback."
                    self._logger.warn(message[i])
                else:
                    message[i] = f"Ignored setting valid parameter '{parameters[i].name}' due to atomic rollback."
                    self._logger.warn(message[i])

        if trigger_deflection_jobs:
            self._logger.debug("Triggering deflection jobs")
            self._timer.reset()

        # construct global result
        result = SetParametersResult()
        result.successful = all(success)
        if len(parameters) == 1:
            result.reason = message[-1]
        else:
            result.reason = str(message)
        info = {'success': success, 'message': message}
        self._logger.debug(f"Parameter change finished: {info}")

        # shutdown after rejected initialization
        if not result.successful:
            rejections = [message[i] for i in range(len(parameters)) if not success[i] and not deflected[i] and not parameters[i].name in self._parameters]
            if len(rejections) > 0:
                log = f"Shutting down node after failed parameter initialization: {rejections if len(rejections) > 1 else rejections[0]}"
                self._logger.error(log)
                raise SelfShutdown(log)

        return result

class ParameterContainer(types.SimpleNamespace):
    """
    Container for storing and managing named parameters.

    This class wraps a SimpleNamespace to store parameters.
    Parameters can be added, updated, or removed using standard Python attribute syntax.
    Access and modification are synchronized using a reentrant lock to ensure thread safety.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super().__setattr__('_lock', threading.RLock())

    def __getitem__(self, key):
        if key == '_lock':
            return super().__getattribute__(key)
        with self._lock:
            return super().__getattribute__(key)

    def __getattribute__(self, name):
        if name == '_lock':
            return super().__getattribute__(name)
        with self._lock:
            return super().__getattribute__(name)

    def __setattr__(self, name, value):
        with self._lock:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        with self._lock:
            super().__delattr__(name)

    def __contains__(self, name):
        with self._lock:
            return name in self.__dict__

    def __repr__(self):
        with self._lock:
            d = f"{self.get()}"[1:-1]
        return f"<ParameterContainer({d})>"

    def get(self):
        """
        Return a deep copy of all parameters as a dictionary.

        Returns:
            dict: A deep copy of the parameter names and their values.
        """
        with self._lock:
            d = self.__dict__.copy()
            del d['_lock']
            return copy.deepcopy(d)

    def set_parameter(self, name, value):
        """
        Set a parameter with the given name and value.

        Args:
            name (str): Parameter name.
            value: Parameter value.
        """
        assert name != "_lock"
        with self._lock:
            setattr(self, name, value)

    def remove_parameter(self, name):
        """
        Remove a parameter by name.

        Args:
            name (str): Parameter name to remove.

        Raises:
            AttributeError: If the parameter does not exist.
        """
        assert name != "_lock"
        with self._lock:
            if hasattr(self, name):
                delattr(self, name)
            else:
                raise AttributeError(f"Parameter '{name}' not found.")

class ParameterContainerProxy:
    """
    Read-only proxy for accessing parameters in a ParameterContainer.

    It allows to read parameters via attribute access (e.g., `proxy.param_name`),
    but prevents any modification or deletion of parameters.
    Intended for safe and fast sharing of parameters.
    """
    def __init__(self, parameter_container):
        assert_type_value(
            obj=parameter_container,
            type_or_value=ParameterContainer,
            name="argument 'parameter_container'"
        )
        self._parameters = parameter_container

    def __getitem__(self, key):
        with self._parameters._lock:
            return getattr(self._parameters, key)

    def __getattr__(self, name):
        with self._parameters._lock:
            return getattr(self._parameters, name)

    def __setattr__(self, name, value):
        if name == "_parameters":
            super().__setattr__(name, value)
        else:
            raise AttributeError("Cannot modify attributes of ParameterContainerProxy")

    def __delattr__(self, name):
        raise AttributeError("Cannot delete attributes of ParameterContainerProxy")

    def __contains__(self, name):
        with self._parameters._lock:
            return name in self._parameters.__dict__

    def __repr__(self):
        with self._parameters._lock:
            d = f"{self.get()}"[1:-1]
        return f"<ParameterContainerProxy({d})>"

    def get(self):
        """
        Return a deep copy of all parameters as a dictionary.

        Returns:
            dict: A deep copy of the parameter names and their values.
        """
        return self._parameters.get()
