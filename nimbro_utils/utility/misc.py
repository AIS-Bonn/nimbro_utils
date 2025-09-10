#!/usr/bin/env python3

import os
import copy
import time
import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    import json
    ORJSON_AVAILABLE = False

import rclpy
from rclpy.impl.rcutils_logger import RcutilsLogger
import builtin_interfaces.msg
from ament_index_python.packages import get_package_prefix

escape = {
    # ANSI escape codes for terminal text formatting, coloring, and control.

    # Attributes:
    #     Text Colors:
    #         - Standard: black, red, green, yellow, blue, magenta, cyan, white, gray
    #         - Dark: darkred, darkgreen, darkyellow, darkblue, darkmagenta, darkcyan, darkgray

    #     Background Colors:
    #         - Standard: bg_black, bg_red, bg_green, bg_yellow, bg_blue,
    #                     bg_magenta, bg_cyan, bg_white, bg_gray
    #         - Dark: bg_darkred, bg_darkgreen, bg_darkyellow, bg_darkblue,
    #                 bg_darkmagenta, bg_darkcyan, bg_darkgray

    #     Text Styles:
    #         - bold: Bold text
    #         - dim: Dim/faint text
    #         - italic: Italic text (may not be supported everywhere)
    #         - underline: Underlined text
    #         - blink: Blinking text
    #         - invert: Reverses foreground and background
    #         - hidden: Invisible text (still selectable)
    #         - strikethrough: Text with a line through it

    #     Miscellaneous Controls:
    #         - end: Reset all styles and colors to default
    #         - clear_line: Clears the current terminal line
    #         - clear_screen: Clears the entire terminal screen
    #         - carriage_return: Moves cursor to beginning of the line
    #         - bell: Triggers a terminal beep
    #         - cursor_hide: Hides the cursor
    #         - cursor_show: Shows the cursor

    # Usage:
    #     print(f"{escape['red']}Error:{escape['end']} Something went wrong.")
    #     print(f"{escape['bold']}{escape['green']}Success!{escape['end']}")
    #     print(f"{escape['clear_line']}{escape['carriage_return']}Overwriting this line...")

    # Foreground (text) colors
    'black': "\033[30m",
    'red': "\033[91m",
    'green': "\033[92m",
    'yellow': "\033[93m",
    'blue': "\033[94m",
    'magenta': "\033[95m",
    'cyan': "\033[96m",
    'white': "\033[97m",
    'gray': "\033[37m",
    'darkgray': "\033[90m",
    'darkred': "\033[31m",
    'darkgreen': "\033[32m",
    'darkyellow': "\033[33m",
    'darkblue': "\033[34m",
    'darkmagenta': "\033[35m",
    'darkcyan': "\033[36m",

    # Background colors
    'bg_black': "\033[40m",
    'bg_red': "\033[101m",
    'bg_green': "\033[102m",
    'bg_yellow': "\033[103m",
    'bg_blue': "\033[104m",
    'bg_magenta': "\033[105m",
    'bg_cyan': "\033[106m",
    'bg_white': "\033[107m",
    'bg_gray': "\033[47m",
    'bg_darkgray': "\033[100m",
    'bg_darkred': "\033[41m",
    'bg_darkgreen': "\033[42m",
    'bg_darkyellow': "\033[43m",
    'bg_darkblue': "\033[44m",
    'bg_darkmagenta': "\033[45m",
    'bg_darkcyan': "\033[46m",

    # Text styles
    'bold': "\033[1m",
    'dim': "\033[2m",
    'italic': "\033[3m",
    'underline': "\033[4m",
    'blink': "\033[5m",
    'invert': "\033[7m",
    'hidden': "\033[8m",
    'strikethrough': "\033[9m",

    # Miscellaneous controls
    'end': "\033[0m",
    'clear_line': "\033[2K",
    'clear_screen': "\033[2J",
    'carriage_return': "\r",
    'bell': "\a",
    'cursor_hide': "\033[?25l",
    'cursor_show': "\033[?25h",
}

def assert_type_value(obj, type_or_value, *, match_types=True, match_inherited_types=True, match_types_as_values=False, name="object", text=None, logger=None):
    """
    Validates that an object is of a specified type or equals a specified value.

    Parameters:
        obj (object): The object to validate.
        type_or_value (type | value | list[type | value] | dict_keys[type | value]):
            A type, value, or list of types/values that `obj` is checked against.
        match_types (bool, optional):
            If True, object types are matched against provided types. Defaults to True.
        match_inherited_types (bool, optional):
            If True, uses isinstance() for type matching; if False, requires exact type equality. Ignored if match_types is False. Defaults to True.
        match_types_as_values (bool, optional):
            If True, types in `type_or_value` are also accepted as values, even when type matching is disabled. Defaults to True.
        name (str, optional):
            Name of the object, used in error messages. Defaults to "object".
        text (str | None, optional):
            Custom assertion message to override the default one. Defaults to None.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs default assertion message and `text` (if set) before raising assertion. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid or `obj` does not match any of the specified types or values.

    Notes:
        - `None` in `type_or_value` is treated always treated as `type(None)` and bypasses `match_types`.
        - Boolean values are never matched as integers (i.e., True/False will not satisfy an int type).
        - Types in `type_or_value` are counted as values only when `match_types_as_values` is True, regardless of `match_types`.
        - Long string representations of `obj` are truncated in messages for readability.
        - If `logger` is provided and validation fails, both the default error message and `text` (if set) are logged.
    """
    # parse arguments

    assert isinstance(obj, object), f"Expected type of argument 'obj' to be 'object', but it is '{type(obj).__name__}'."
    assert isinstance(match_types, bool), f"Expected type of argument 'match_types' to be 'bool', but it is '{type(match_types).__name__}'."
    assert isinstance(match_inherited_types, bool), f"Expected type of argument 'match_inherited_types' to be 'bool', but it is '{type(match_inherited_types).__name__}'."
    assert isinstance(match_types_as_values, bool), f"Expected type of argument 'match_types_as_values' to be 'bool', but it is '{type(match_types_as_values).__name__}'."
    assert isinstance(name, str), f"Expected type of argument 'name' to be 'str', but it is '{type(name).__name__}'."
    assert text is None or isinstance(text, str), f"Expected type of argument 'text' to be in {[type(None), str]}, but it is '{type(text).__name__}'."
    from nimbro_utils.node_extensions.logger import Logger
    assert logger is None or isinstance(logger, (RcutilsLogger, Logger)), f"Expected type of argument 'logger' to be in {[RcutilsLogger, Logger]}, but it is '{type(logger).__name__}'."

    max_value_length = 50
    if isinstance(obj, type):
        value = obj.__name__
    elif len(str(obj)) > max_value_length:
        value = str(obj)[:max_value_length] + " ..."
    else:
        value = str(obj)

    if isinstance(type_or_value, type({}.keys())):
        type_or_value = list(type_or_value)
    elif not isinstance(type_or_value, (tuple, list)):
        type_or_value = [type_or_value]
    assert len(type_or_value) > 0, "Expected argument 'type_or_value' to be a value, a type, or a list thereof, but it is an empty list."

    # collect matching targets
    types_list, type_names, values_list, value_names, value_types = [], [], [], [], []
    for item in type_or_value:
        if item is None or isinstance(item, type(None)):
            type_names.append("NoneType")
            values_list.append(None)
        elif isinstance(item, type):
            assert match_types or match_types_as_values, "Expected either 'match_types' or 'match_types_as_values' to be True when 'type_or_value' contains an element of type 'type'."
            if match_types:
                types_list.append(item)
                type_names.append(item.__name__)
            if match_types_as_values:
                values_list.append(item)
                value_names.append(item.__name__)
                value_types.append("type")
        else:
            values_list.append(item)
            value_names.append(f"{item}")
            value_types.append(type(item).__name__)

    # check validity and generate error text

    def _is_valid_type(o):
        for t in types_list:
            if t is int and isinstance(o, bool):
                continue
            if match_inherited_types:
                if isinstance(o, t):
                    return True
            else:
                if type(o) is t:
                    return True
        return False

    def _typed_in(value, collection):
        for item in collection:
            if type(value) is type(item):
                try:
                    if value == item:
                        return True
                except Exception:
                    pass
        return False

    valid_type = match_types and _is_valid_type(obj)
    valid_value = _typed_in(obj, values_list)
    valid = valid_type or valid_value

    if len(type_names) > 0 and len(value_names) > 0:
        if not valid:
            if len(type_names) == 1:
                if len(value_names) == 1:
                    _text = f"Expected type of {name} to be '{type_names[0]}', or its value to be '{value_names[0]}' of type '{value_types[0]}', but it is '{value}' of type '{type(obj).__name__}'."
                else:
                    _text = f"Expected type of {name} to be '{type_names[0]}', or its value to be in {value_names} with types {value_types}, but it is '{value}' of type '{type(obj).__name__}'."
            else:
                if len(value_names) == 1:
                    _text = f"Expected type of {name} to be in {type_names}, or its value to be '{value_names[0]}' of type '{value_types[0]}', but it is '{value}' of type '{type(obj).__name__}'."
                else:
                    _text = f"Expected type of {name} to be in {type_names}, or its value to be in {value_names} with types {value_types}, but it is '{value}' of type '{type(obj).__name__}'."
    elif len(type_names) > 0:
        if not valid:
            if len(type_names) == 1:
                _text = f"Expected type of {name} to be '{type_names[0]}', but it is '{type(obj).__name__}'."
            else:
                _text = f"Expected type of {name} to be in {type_names}, but it is '{type(obj).__name__}'."
    else:
        if not valid:
            if len(value_names) == 1:
                _text = f"Expected value of {name} to be '{value_names[0]}' of type '{value_types[0]}', but it is '{value}' of type '{type(obj).__name__}'."
            else:
                _text = f"Expected value of {name} to be in {value_names} with types {value_types}, but it is '{value}' of type '{type(obj).__name__}'."

    # log
    if not valid and logger is not None:
        logger.error(_text)
        if text is not None:
            logger.error(text)

    # assert
    assert valid, _text if text is None else text

def assert_attribute(obj, attribute, *, exists=True, name="object", text=None, logger=None):
    """
    Validates that an object has (or has not) a (possibly nested) attribute.

    Parameters:
        obj (object): The object to validate.
        exists (bool, optional): Determines if the attribute must exist or must not exist. Default to True.
        attribute (str): Name of the attribute relative to `obj` to check for. Dot notation (e.g., "a.b.c") is supported for nested attributes.
        name (str, optional): Name of the object, used in error messages. Defaults to "object".
        text (str | None, optional): Custom assertion message to override the default one. Defaults to None.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs default assertion message and `text` (if set) before raising assertion. Defaults to None.

    Returns:
        Any: The value of the final attribute in the chain.

    Raises:
        AssertionError: If input arguments are invalid or any attribute in the chain is missing.

    Notes:
        - Nested attributes are resolved recursively using dot notation.
        - If `logger` is provided and validation fails, both the default error message and `text` (if set) are logged.
    """
    # parse arguments
    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'", logger=None)
    assert_type_value(obj=obj, type_or_value=object, name="argument 'obj'", logger=logger)
    assert_type_value(obj=attribute, type_or_value=str, name="argument 'attribute'", logger=logger)
    assert_type_value(obj=exists, type_or_value=bool, name="argument 'exists'", logger=logger)
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'", logger=logger)
    assert_type_value(obj=text, type_or_value=[str, None], name="argument 'text'", logger=logger)

    # validate attribute
    has_attribute = hasattr(obj, attribute)
    if not has_attribute:
        _text = f"Expected {name} to have the attribute '{attribute}'."

    # validate attribute
    parts = attribute.split(".")
    current = obj
    has_attribute = True
    for i, part in enumerate(parts):
        if not hasattr(current, part):
            has_attribute = False
            if i == 0:
                if len(parts) > 1:
                    _text = f"Expected {name} to have the attribute '{attribute}', but it does not have the attribute '{part}'."
                else:
                    _text = f"Expected {name} to have the attribute '{attribute}'."
            else:
                _text = f"Expected {name} to have the attribute '{attribute}', but the attribute '{parts[i - 1]}' does not have the attribute '{part}'."
            break
        current = getattr(current, part)

    # exists
    if not exists:
        if has_attribute:
            _text = f"Expected {name} to not have the attribute '{attribute}'."
            has_attribute = False
        else:
            has_attribute = True

    # log
    if not has_attribute and logger is not None:
        logger.error(_text)
        if text is not None:
            logger.error(text)

    # assert
    assert has_attribute, _text if text is None else text

    # return attribute
    return (current if exists else None) if has_attribute else None

def assert_keys(obj, keys, mode="whitelist", *, name="dictionary", text=None, logger=None):
    """
    Validates that a dictionary contains or omits specific keys, depending on the selected mode.

    Parameters:
        obj (dict): The dictionary to validate.
        keys (set | list | tuple | dict_keys): The collection of keys to check against.
        mode (str, optional): The validation mode. Must be one of:
            - "match": All and only the specified keys must be present in `obj`.
            - "whitelist": The specified keys are permitted to be present in `obj`.
            - "required": The specified keys must be present, but extra keys are allowed.
            - "blacklist": The specified keys must not be present.
            Defaults to "whitelist".
        name (str, optional): A descriptive name for the dictionary, used in error messages. Defaults to "dictionary".
        text (str | None, optional): Custom assertion message to override the default one. Defaults to None.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs default assertion message and `text` (if set) before raising assertion. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid or the dictionary violates the key constraints defined by `mode`.

    Notes:
        - In "match" mode, `obj` must contain exactly the keys in `keys`.
        - In "whitelist" mode, `obj` must not contain any keys that are not in `keys`, but does not require any of them.
        - In "required" mode, `obj` must contain at least the keys in `keys`, but extra keys are allowed.
        - In "blacklist" mode, `obj` must not contain any of the keys in `keys`.
        - If `logger` is provided and validation fails, both the default error message and `text` (if set) are logged.
    """
    # parse arguments
    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'", logger=None)
    assert_type_value(obj=obj, type_or_value=dict, name="argument 'obj'", logger=logger)
    assert_type_value(obj=keys, type_or_value=[set, list, tuple, type({}.keys())], name="argument 'keys'", logger=logger)
    assert_type_value(obj=mode, type_or_value=["match", "whitelist", "blacklist", "required"], name="argument 'mode'", logger=logger)
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'", logger=logger)
    assert_type_value(obj=text, type_or_value=[str, None], name="argument 'text'", logger=logger)

    # missing keys
    if mode in ["match", "required"]:
        missing_keys = []
        for key in keys:
            if key not in obj:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            if len(missing_keys) == len(keys):
                _text = f"Expected {name} to contain the key{(" '" + str(list(keys)[0]) + "'") if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {'all of them' if len(missing_keys) > 1 else 'it'}." # noqa
            else:
                _text = f"Expected {name} to contain the key{(" '" + str(list(keys)[0]) + "'") if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {str(missing_keys) if len(missing_keys) > 1 else ("'" + str(missing_keys[0]) + "'")}."
            if logger is not None:
                logger.error(_text)
                if text is not None:
                    logger.error(text)
            assert len(missing_keys) == 0, _text if text is None else text

    # excessive keys
    if mode in ["match", "whitelist"]:
        excessive_keys = []
        for key in obj:
            if key not in keys:
                excessive_keys.append(key)
        if len(excessive_keys) > 0:
            if len(excessive_keys) == len(keys):
                _text = f"Expected {name} to contain the key{(" '" + str(list(keys)[0]) + "'") if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {'all of them' if len(excessive_keys) > 1 else 'it'}."
            else:
                _text = f"Expected {name} to contain only the key{(" '" + str(list(keys)[0]) + "'") if len(keys) == 1 else ('s ' + str(list(keys)))} but it contains {str(excessive_keys) if len(excessive_keys) > 1 else ("'" + str(excessive_keys[0]) + "'")}."
            if logger is not None:
                logger.error(_text)
                if text is not None:
                    logger.error(text)
            assert len(excessive_keys) == 0, _text if text is None else text

    # forbidden keys
    elif mode == "blacklist":
        forbidden_keys = []
        for key in obj:
            if key in keys:
                forbidden_keys.append(key)
        if len(forbidden_keys) > 0:
            if len(forbidden_keys) == len(keys):
                _text = f"Expected {name} to contain the key{(" '" + str(list(keys)[0]) + "'") if len(keys) == 1 else ('s ' + str(list(keys)))} but it misses {'all of them' if len(forbidden_keys) > 1 else 'it'}."
            else:
                _text = f"Expected {name} to not contain the key{(" '" + str(list(keys)[0]) + "'") if len(keys) == 1 else ('s ' + str(list(keys)))} but it {('contains' + str(forbidden_keys)) if len(keys) > 1 else 'does'}."
            if logger is not None:
                logger.error(_text)
                if text is not None:
                    logger.error(text)
            assert len(forbidden_keys) == 0, _text if text is None else text

def read_json(file_path, name="file", logger=None):
    """
    Read and decode a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        name (str, optional): Descriptive name for logging. Defaults to "file".
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid (excluding invalid file path or file containing invalid JSON).

    Returns:
        tuple[bool, str, Any]: Success flag, status message, decoded JSON object if success or None if not success.

    Notes:
        - If available, this function uses the faster 'orjson' module; otherwise, it falls back to the standard 'json' module.

    """
    # parse arguments
    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'", logger=None)
    assert_type_value(obj=file_path, type_or_value=str, name="argument 'file_path'", logger=logger)
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'", logger=logger)

    # read and decode file

    tic = time.perf_counter()

    if not os.path.exists(file_path):
        message = f"Expected path '{file_path}' to exist."
        if logger is not None:
            logger.error(message)
        assert os.path.exists(file_path), message
    elif not os.path.isfile(file_path):
        message = f"Expected path '{file_path}' to be a file."
        if logger is not None:
            logger.error(message)
        assert os.path.isfile(file_path), message

    if logger is not None:
        logger.debug(f"Reading {name} '{file_path}'")

    try:
        if ORJSON_AVAILABLE:
            with open(file_path, "rb") as f:
                json_object = orjson.loads(f.read())
        else:
            if logger is not None:
                logger.warn(f"Using slow 'json' module to read {name}. Install 'orjson' to speed this up!", once=True)
            with open(file_path, 'r') as f:
                json_object = json.load(f)
    except Exception as e:
        success = False
        message = f"Failed to read {name} '{file_path}': {repr(e)}"
        if logger is not None:
            logger.error(message)
        json_object = None
    else:
        success = True
        message = f"Read {name} '{file_path}' in '{time.perf_counter() - tic:.3f}s'."
        if logger is not None:
            logger.debug(message[:-1])

    return success, message, json_object

def write_json(file_path, json_object, indent=True, name="file", logger=None):
    """
    Encode and write a JSON object.

    Args:
        file_path (str): Destination path for the JSON file.
        json_object (any): Object to serialize to JSON.
        indent (bool, optional): Pretty-print with indentation (2 spaces) when True. Defaults to True.
        name (str, optional): Descriptive name for logging. Defaults to "file".
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid (excluding invalid file path or invalid JSON object).

    Returns:
        tuple[bool, str]: Success flag, status message.

    Notes:
        - If available, this function uses the faster 'orjson' module; otherwise, it falls back to the standard 'json' module.
    """
    # parse arguments
    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'", logger=None)
    assert_type_value(obj=file_path, type_or_value=str, name="argument 'file_path'", logger=logger)
    assert_type_value(obj=indent, type_or_value=bool, name="argument 'indent'", logger=logger)
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'", logger=logger)

    # encode and write file

    tic = time.perf_counter()
    file_path = os.path.abspath(file_path)

    if logger is not None:
        logger.debug(f"Writing {name} '{file_path}'")

    target_folder = os.path.dirname(file_path)
    if not os.path.exists(target_folder):
        if logger is not None:
            logger.debug(f"Creating directory '{target_folder}'")
        try:
            os.makedirs(target_folder)
        except Exception as e:
            success = False
            message = f"Failed to create directory '{target_folder}': {repr(e)}"
            if logger is not None:
                logger.error(message)
            return success, message
    elif not os.path.isdir(target_folder):
        if logger is not None:
            message = f"Expected path '{target_folder}' to either not exist or be a directory."
            logger.error(message)
        assert os.path.isdir(target_folder), message

    try:
        if ORJSON_AVAILABLE:
            with open(file_path, "wb") as f:
                if indent:
                    f.write(orjson.dumps(json_object, option=orjson.OPT_INDENT_2))
                else:
                    f.write(orjson.dumps(json_object))
        else:
            if logger is not None:
                logger.warn(f"Using slow 'json' module to write {name}. Install 'orjson' to speed this up!", once=True)
            with open(file_path, 'w') as f:
                json.dump(json_object, f, indent=2 if indent else None)
    except Exception as e:
        success = False
        message = f"Failed to write {name} '{file_path}': {repr(e)}"
        if logger is not None:
            logger.error(message)
    else:
        success = True
        message = f"Written {name} '{file_path}' in '{time.perf_counter() - tic:.3f}s'."
        if logger is not None:
            logger.debug(message[:-1])

    return success, message

def update_dict(old_dict, new_dict=None, deepcopy=False, key_name=None, logger=None, info=True, debug=False):
    """
    Update a dictionary by merging keys from an old dictionary into a new dictionary, filling in missing keys.

    Parameters:
        old_dict (dict): The original dictionary containing default key-value pairs.
        new_dict (dict | None): The dictionary to update, or None to create a new one. Defaults to None.
        deepcopy (bool, optional): If True, new_dict is deep copied before inserting missing items from old_dict. Defaults to False
        key_name (str | None, optional): Name to use for keys in log messages. Defaults to "key" if None.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs updates bases on `info` and `debug`. Defaults to None.
        info (bool, optional): If True and logger is set, logs of key updates use severity INFO instead of DEBUG. Defaults to True.
        debug (bool, optional): If True and logger is set, keys that are not updated are logged using severity DEBUG. Defaults to False.
    Returns:
        dict: A new dictionary containing all keys from `new_dict` (or an empty dict if None),
        updated with keys from `old_dict` where missing.

    Raises:
        AssertionError: If input arguments are invalid.

    Notes:
        - If `new_dict` is None, an empty dictionary is created.
        - The returned dictionary is a deep copy to avoid modifying the input dictionaries.
        - If a logger is provided, debug messages are logged when keys are initialized or updated.
        - The `key_name` parameter customizes the descriptor used in log messages for keys.
    """
    # parse arguments

    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(logger, [RcutilsLogger, Logger, None], name="argument 'logger'")
    assert_type_value(old_dict, dict, name="argument 'old_dict'", logger=logger)
    assert_type_value(new_dict, [dict, None], name="argument 'new_dict'", logger=logger)
    assert_type_value(deepcopy, bool, name="argument 'deepcopy'", logger=logger)
    assert_type_value(key_name, [str, None], name="argument 'key_name'", logger=logger)
    assert_type_value(info, bool, name="argument 'info'", logger=logger)
    assert_type_value(debug, bool, name="argument 'debug'", logger=logger)

    if key_name is None:
        key_name = "key"
    if new_dict is None:
        new_dict = {}
    elif deepcopy:
        new_dict = copy.deepcopy(new_dict)

    # log
    if logger is not None:
        for key in dict.fromkeys(list(old_dict.keys()) + list(new_dict.keys())):
            if key in old_dict and key in new_dict:
                if old_dict[key] == new_dict[key]:
                    if debug:
                        logger.debug(f"Ignoring redundant update of {key_name} '{key}' set to '{new_dict[key]}'")
                else:
                    if info:
                        logger.info(f"Updating {key_name} '{key}' from '{old_dict[key]}' to '{new_dict[key]}'")
                    else:
                        logger.debug(f"Updating {key_name} '{key}' from '{old_dict[key]}' to '{new_dict[key]}'")
            elif key in old_dict:
                if debug:
                    logger.debug(f"Maintaining {key_name} '{key}' set to '{old_dict[key]}'")
            else:
                if info:
                    logger.info(f"Initializing {key_name} '{key}' to '{new_dict[key]}'")
                else:
                    logger.debug(f"Initializing {key_name} '{key}' to '{new_dict[key]}'")

    # update
    for key in old_dict:
        if key not in new_dict:
            new_dict[key] = old_dict[key]

    return new_dict

def count_duplicates(iterable, include_unique=False):
    """
    Count occurrences of items in an iterable and return duplicates or all counts.

    Args:
        iterable (iterable): Collection of hashable items to count.
        include_unique (bool, optional): If True, include items that occur only once in the result. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        dict: Mapping of items to their occurrence counts.
    """
    assert iter(iterable), "Expected value of argument 'iterable' to be iterable."
    assert_type_value(obj=include_unique, type_or_value=bool, name="argument 'include_unique'")

    count_dict = {}
    for item in iterable:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    if include_unique:
        return count_dict
    else:
        return {key: value for key, value in count_dict.items() if value > 1}

def start_jobs(jobs, job_args=None, timeout=None, max_workers=None, name="job", level="normal", logger=None):
    """
    Run multiple callable jobs in parallel using a thread pool.

    Each job is executed concurrently. The function waits for at most `timeout` seconds
    and collects the result of each job that completes in time. If a job raises an exception,
    it is caught and included in the result. Jobs that don't finish within the timeout
    are reported as incomplete and continue running in the background.

    Args:
        jobs (list[Callable]): A list of no-argument functions or bound methods to execute.
        job_args (list[None | dict | tuple], optional): Optional list of arguments per job.
            Must be None or a list of the same length as `jobs`.
            List values can be None (no args), dict (kwargs only),
            or tuple (args, kwargs) where args is tuple or list and kwargs is dict.
            Defaults to None.
        timeout (float | int | None, optional): Maximum number of seconds to wait for all jobs to complete.
            If None, waits indefinitely. Defaults to None.
        max_workers (int | None, optional): Maximum number of worker threads. If None, it defaults to
            the number of processors on the machine, multiplied by 5. Defaults to None.
        name (str, optional): Descriptive name for logging. Defaults to "job".
        level (str, optional): Use 'normal' to use regular logger severities and 'debug' to use severity debug only. Defaults to 'normal'.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        list[tuple[bool, str, Any]]:
            A list of tuples, one per job:
                - success (bool): True if the job completed successfully within the timeout,
                  False otherwise (including exceptions and timeout).
                - message (str): A descriptive message about the job result.
                - result (Any | None): The return value of the job if it succeeded, otherwise None.
    """
    # parse arguments
    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'")
    assert_type_value(obj=jobs, type_or_value=list, name="argument 'jobs'")
    for item in jobs:
        if not callable(item):
            message = "Expected argument 'jobs' to contain callable items."
            if logger is not None:
                logger.error(message)
            assert callable(item), message
    assert_type_value(obj=job_args, type_or_value=[list, None], name="argument 'job_args'")
    if job_args is not None:
        if len(job_args) != len(jobs):
            message = f"Expected argument 'job_args' provided as list match the number of jobs '{len(jobs)}' but got '{len(job_args)}'."
            if logger is not None:
                logger.error(message)
            assert len(job_args) == len(jobs), message
        for i, arg in enumerate(job_args):
            assert_type_value(obj=arg, type_or_value=[dict, tuple, None], name="all items in argument 'job_args'")
            if isinstance(arg, tuple):
                if not (len(arg) == 2 and isinstance(arg[0], (list, tuple)) and isinstance(arg[1], dict)):
                    message = "Expected argument 'job_args' provided as list to contain only values that are None, dict, or tuple (args, kwargs), where args is a tuple or list, and kwargs is a dict."
                    if logger is not None:
                        logger.error(message)
                    assert len(arg) == 2 and isinstance(arg[0], (list, tuple)) and isinstance(arg[1], dict), message
    assert_type_value(obj=timeout, type_or_value=[float, int, None], name="argument 'timeout'")
    assert_type_value(obj=max_workers, type_or_value=[int, None], name="argument 'max_workers'")
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
    assert_type_value(obj=level, type_or_value=["normal", "debug"], name="argument 'level'")

    if len(jobs) == 0:
        return []

    # start jobs

    if logger is not None:
        message = f"Starting '{len(jobs)}' {name}{'' if len(jobs) == 1 else 's'} (timeout={timeout}{'' if timeout is None else 's'}, max_workers={max_workers})"
        if level == "normal":
            logger.info(message)
        else:
            logger.debug(message)

    results = [None] * len(jobs)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    def wrap(func, arg_spec):
        if arg_spec is None:
            return lambda: func()
        if isinstance(arg_spec, dict):
            return lambda: func(**arg_spec)
        args, kwargs = arg_spec
        return lambda: func(*args, **kwargs)

    futures = {
        executor.submit(wrap(jobs[i], None if job_args is None else job_args[i])): i
        for i in range(len(jobs))
    }

    # collect results

    start_time = time.perf_counter()
    completed = 0
    try:
        for future in as_completed(futures, timeout=timeout):
            idx = futures[future]
            try:
                value = future.result()
            except Exception as e:
                duration = time.perf_counter() - start_time
                if logger is not None:
                    message = f"{name.capitalize()} '{idx}' raised an exception after '{duration:.2f}s': {repr(e)}"
                    if level == "normal":
                        logger.error(message)
                    else:
                        logger.debug(message)
                results[idx] = (False, f"{name.capitalize()} raised an exception after '{duration:.2f}s': {repr(e)}", None)
            else:
                duration = time.perf_counter() - start_time
                if logger is not None:
                    message = f"{name.capitalize()} '{idx}' completed after '{duration:.2f}s'."
                    if level == "normal":
                        logger.info(message)
                    else:
                        logger.debug(message)
                results[idx] = (True, f"{name.capitalize()} completed after '{duration:.2f}s'.", value)
                completed += 1
    except TimeoutError:
        duration = time.perf_counter() - start_time
        for idx, result in enumerate(results):
            if result is None:
                if logger is not None:
                    message = f"{name.capitalize()} '{idx}' did not finish after timeout of '{duration:.2f}s'."
                    if level == "normal":
                        logger.warn(message)
                    else:
                        logger.debug(message)
                results[idx] = (False, f"{name.capitalize()} did not finish after timeout of '{duration:.2f}s'.", None)
        if logger is not None:
            message = f"Successfully completed '{completed}' out of '{len(results)}' {name}{'' if len(results) == 1 else 's'} after timeout of '{duration:.2f}s'."
            if level == "normal":
                logger.info(message)
            else:
                logger.debug(message)
    else:
        if logger is not None:
            duration = time.perf_counter() - start_time
            if completed == len(results):
                message = f"Successfully completed all '{len(results)}' {name}s after '{duration:.2f}s'."
                if level == "normal":
                    logger.info(message)
                else:
                    logger.debug(message)
            else:
                message = f"Successfully completed '{completed}' out of '{len(results)}' {name}{'' if len(results) == 1 else 's'} after '{duration:.2f}s'."
                if level == "normal":
                    logger.info(message)
                else:
                    logger.debug(message)
    finally:
        executor.shutdown(wait=False)

    return results

def try_callback(callback, mode="forward", logger=None):
    """
    Wraps a callback function in a try/except block and handles it in a predefined behavior.

    Args:
        callback (callable): The callback function to be wrapped.
        mode (str, optional): Defines the behavior of this wrapper. Any of:
            - 'graceful': Turn exception into `SelfShutdown` without any additional message or traceback.
            - 'swallow': Swallow exception after logging it as debug message without traceback.
            - 'error': Swallow exception after logging it as error message without traceback.
            - 'traceback': Swallow exception after logging it as error message with traceback.
            - 'forward': Raise exception as is after logging it as error message with traceback.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        callable: The wrapped callback function.

    Notes:
        - If a `SelfShutdown` is raised inside the callback, this wrapper
            forwards it directly without doing anything regardless of the mode set.
        - Example: self.create_subscription(..., callback=try_callback(self, self.my_callback), ...)
    """
    # parse arguments
    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'")
    if not callable(callback):
        message = "Expected argument 'callback' to be callable."
        if logger is not None:
            logger.error(message)
        assert callable(callback), message
    assert_type_value(obj=mode, type_or_value=["graceful", "swallow", "error", "traceback", "forward"], name="argument 'mode'", logger=logger)

    def callback_wrapper(*args, **kwargs):
        from nimbro_utils.utility.node import SelfShutdown
        try:
            return callback(*args, **kwargs)
        except SelfShutdown as e:
            raise SelfShutdown from e
        except Exception as e:
            if mode == "graceful":
                raise SelfShutdown(f"{e}") from e
            elif mode == "swallow":
                if logger is not None:
                    logger.debug(f"{repr(e)}")
                else:
                    print(f"{repr(e)}")
            elif mode == "error":
                if logger is not None:
                    logger.error(f"{repr(e)}")
                else:
                    print(f"{escape['red']}{repr(e)}{escape['end']}")
            elif mode == "traceback":
                if logger is not None:
                    logger.error(f"{repr(e)}\n{traceback.format_exc()}")
                else:
                    print(f"{escape['red']}{repr(e)}\n{traceback.format_exc()}{escape['end']}")
            elif mode == "forward":
                if logger is not None:
                    logger.error(f"{repr(e)}\n{traceback.format_exc()}")
                else:
                    print(f"{escape['red']}{repr(e)}\n{traceback.format_exc()}{escape['end']}")
                raise
            else:
                raise NotImplementedError(f"Unknown 'mode' value '{mode}'.")

    return callback_wrapper

def convert_stamp(stamp, target_format="iso"):
    """
    Convert a timestamp from one format to another (UTC).

    Args:
        stamp (float | int | str | datetime.datetime | builtin_interfaces.msg.Time | rclpy.time.Time):
            The input timestamp to convert. Supported formats:
            - float | int: UNIX timestamp in seconds (e.g., 1721030400.0).
            - str: ISO 8601 formatted string (e.g., '2025-07-15T12:34:56').
            - datetime.datetime: Python datetime object.
            - rclpy.time.Time: ROS2 time object.
            - builtin_interfaces.msg.Time: ROS2 builtin timestamp message.

        target_format (str, optional): Desired output format. Must be one of:
            - 'seconds': Returns a float representing seconds since epoch.
            - 'nanoseconds': Returns an integer representing nanoseconds since epoch.
            - 'iso': Returns an ISO 8601 formatted string.
            - 'datetime': Returns a Python datetime.datetime object.
            - 'rclpy': Returns an rclpy.time.Time object.
            - 'msg': Returns a builtin_interfaces.msg.Time message.
            Defaults to 'iso'.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        float | int | str | datetime.datetime | rclpy.time.Time | builtin_interfaces.msg.Time:
            The converted timestamp in the requested format.
    """
    # parse arguments
    assert_type_value(stamp, [float, int, str, datetime.datetime, rclpy.time.Time, builtin_interfaces.msg.Time], name="argument 'stamp'")
    assert_type_value(target_format, ["seconds", "nanoseconds", "iso", "datetime", "rclpy", "msg"], name="argument 'target_format'")

    # convert to seconds
    if isinstance(stamp, (float, int)):
        seconds = stamp
    elif isinstance(stamp, str):
        seconds = datetime.datetime.fromisoformat(stamp).timestamp()
    elif isinstance(stamp, datetime.datetime):
        seconds = (stamp - datetime.datetime(1970, 1, 1)).total_seconds()
    elif isinstance(stamp, rclpy.time.Time):
        seconds = stamp.seconds_nanoseconds()
        seconds = seconds[0] + seconds[1] / 1e9
    elif isinstance(stamp, builtin_interfaces.msg.Time):
        seconds = rclpy.time.Time.from_msg(stamp)
        seconds = seconds.seconds_nanoseconds()
        seconds = seconds[0] + seconds[1] / 1e9
    else:
        raise NotImplementedError(f"Unknown 'stamp' type '{type(stamp).__name__}'.")

    # convert to target format
    if target_format == "seconds":
        result = seconds
    elif target_format == "nanoseconds":
        result = seconds * 1e9
    elif target_format == "iso":
        result = (datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=seconds)).isoformat()
    elif target_format == "datetime":
        result = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=seconds)
    elif target_format == "rclpy":
        result = rclpy.time.Time(seconds=int(seconds), nanoseconds=(seconds % 1) * 1e9)
    elif target_format == "msg":
        result = rclpy.time.Time(seconds=int(seconds), nanoseconds=(seconds % 1) * 1e9).to_msg()
    else:
        raise NotImplementedError(f"Unknown 'target_format' '{target_format}'.")

    # print(f"{stamp} ({type(stamp)})\n{target_format}\n{seconds}\n{result}\n{type(result).__name__}\n")
    return result

def log_lines(text, line_length=150, line_highlight="| ", block_format=False, allow_empty_lines=False, max_lines=100, logger=None, severity=None):
    """
    Splits the given text into wrapped lines and logs or prints them, adding a prefix to all but the first line.
    Long lines are broken at the nearest space before the maximum length when possible. Output is either printed
    or sent to a logger at the specified severity level.

    Args:
        text (str):
            The text to log, which may span multiple lines.
        line_length (int, optional):
            The maximum number of characters per line before wrapping.
            Defaults to 150.
        line_highlight (str, optional):
            A prefix string applied to all lines except the first.
            Defaults to "| ".
        block_format (bool, optional):
            If True, all wrapped lines except the last in a block are justified by inserting
            extra spaces between words so they are exactly `line_length` characters long.
            Defaults to False.
        allow_empty_lines (bool, optional):
            If True, empty lines or lines containing only whitespaces are preserved and logged. Defaults to False.
        max_lines (int, optional):
            The maximum number of lines after which input is cutoff and not logged. Defaults to 100.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            A logger instance for output. If None, output is printed to standard output. Defaults to None.
        severity (int, optional):
            Logging severity level corresponding to Python's logging levels:
            10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=FATAL.
            Ignored if `logger` is None. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid.
    """
    # parse arguments
    from nimbro_utils.node_extensions.logger import Logger
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'")
    assert_type_value(obj=text, type_or_value=str, name="argument 'text'")
    assert_type_value(obj=line_length, type_or_value=int, name="argument 'line_length'")
    line_length = max(line_length, 1)
    assert_type_value(obj=line_highlight, type_or_value=str, name="argument 'line_highlight'")
    if logger is not None:
        assert_type_value(obj=severity, type_or_value=[10, 20, 30, 40, 50], name="argument 'severity'")
    assert_type_value(obj=block_format, type_or_value=bool, name="argument 'block_format'")
    assert_type_value(obj=allow_empty_lines, type_or_value=bool, name="argument 'allow_empty_lines'")
    assert_type_value(obj=max_lines, type_or_value=int, name="argument 'max_lines'")

    # define helpers

    def log_line(line, is_first):
        if not is_first:
            line = f"{line_highlight}{line}"
        if logger is None:
            print(line)
        elif severity == 10:
            logger.debug(line)
        elif severity == 20:
            logger.info(line)
        elif severity == 30:
            logger.warn(line)
        elif severity == 40:
            logger.error(line)
        elif severity == 50:
            logger.fatal(line)

    def justify_line(line, is_first):
        words = line.split()
        if len(words) == 1:
            return line
        total_chars = sum(len(w) for w in words)
        if is_first:
            spaces_needed = line_length - total_chars + len(line_highlight)
        else:
            spaces_needed = line_length - total_chars
        gaps = len(words) - 1
        space_width, extra = divmod(spaces_needed, gaps)
        justified_parts = []
        for i, word in enumerate(words[:-1]):
            justified_parts.append(word)
            justified_parts.append(" " * (space_width + (1 if i < extra else 0)))
        justified_parts.append(words[-1])
        return "".join(justified_parts)

    # split lines
    is_first = True
    num_lines = 0
    for paragraph in text.split("\n"):
        # handle empty or whitespace-only paragraphs
        if paragraph == "" or paragraph.isspace():
            if allow_empty_lines:
                log_line(paragraph, is_first)
                num_lines += 1
                is_first = False
            continue

        # normal non-empty paragraph: wrap and (optionally) justify
        lines = []
        line = paragraph
        while len(line) > line_length:
            break_pos = line.rfind(' ', 0, line_length)
            if break_pos == -1:
                part = line[:line_length]
                line = line[line_length:]
            else:
                part = line[:break_pos]
                line = line[break_pos + 1:]
            lines.append(part)
        lines.append(line)

        if block_format:
            for i in range(len(lines) - 1):
                lines[i] = justify_line(lines[i], is_first and i==0)

        for part in lines:
            log_line(part, is_first)
            num_lines += 1
            is_first = False
            if num_lines >= abs(max_lines):
                log_line("...", is_first)
                return

def get_package_path(name, mode="source"):
    """
    Retrieve the package path for a specified ROS2 package name.

    Args:
        name (str): Name of a n installed ROS2 package.
        mode (str, optional): Specify which path to get to the package.
                              Must be in ['source', 'install', 'build'].
                              Defaults to 'source'.

    Raises:
        AssertionError: If input arguments are invalid.
        RuntimeError: If operation unexpectedly fails.

    Returns:
        str: Requested path.
    """
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
    assert_type_value(obj=mode, type_or_value=["source", "install", "build"], name="argument 'mode'")
    package_path = get_package_prefix(name)
    if mode == "source":
        package_path = package_path.replace("install", "src")
    elif mode == "build":
        package_path = package_path.replace("install", "build")
    if not os.path.exists(package_path):
        raise RuntimeError(f"Expected package path '{package_path}' to exist.")
    elif not os.path.isdir(package_path):
        raise RuntimeError(f"Expected package path '{package_path}' to be a directory.")
    return package_path

def in_jupyter_notebook():
    """
    Detects whether this function is executed from inside a Jupyter notebook.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False
