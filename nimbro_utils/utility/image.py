#!/usr/bin/env python3

import os
import math
import time
import uuid

try:
    import pybase64
    PYBASE64_AVAILABLE = True
except ImportError:
    import base64
    PYBASE64_AVAILABLE = False

import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from rclpy.impl.rcutils_logger import RcutilsLogger

from nimbro_utils.node_extensions.logger import Logger
from nimbro_utils.utility.color import ColorPalette, nimbro
from nimbro_utils.utility.string import is_url, normalize_string
from nimbro_utils.utility.geometry import convert_boxes
from nimbro_utils.utility.misc import assert_type_value, convert_stamp, in_jupyter_notebook, get_package_path

bridge = CvBridge()

IMAGE_ENCODINGS = {
    "16UC1": (np.uint16, 1, "DEPTH"),
    "mono8": (np.uint8, 1, "GRAY"),
    "mono16": (np.uint16, 1, "GRAY"),
    "bgr8": (np.uint8, 3, "BGR"),
    "rgb8": (np.uint8, 3, "RGB"),
    "bgr16": (np.uint16, 3, "BGR"),
    "rgb16": (np.uint16, 3, "RGB")
}

# general

def show_image(image, width=None, is_rgb=False):
    """
    Displays an image using the best available method (Jupyter, OpenCV, or matplotlib).

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The image to be shown. See `convert_image()` for compatibility.
        width (int | None, optional):
            If set, the image is scaled to this width while preserving aspect ratio. Defaults to None.
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (True) or BGR (False) order.
            Defaults to False.

    Raises:
        AssertionError: If arguments are invalid.
        RuntimeError: If no backend for visualization can be found.
    """
    # parse arguments
    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'image'")
    assert_type_value(obj=width, type_or_value=[int, None], name="argument 'width'")
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")

    # convert and scale
    image = convert_image(image, target_format="NumPy", target_encoding=None, is_rgb=is_rgb)
    if width is not None:
        image = resize_image(image, value=width, mode="width", is_rgb=is_rgb)
    image = convert_image(image, target_format="NumPy", target_encoding="bgr8", is_rgb=is_rgb)

    # display

    if in_jupyter_notebook():
        from IPython.display import Image as _Image
        from IPython.display import display
        success, buffer = cv2.imencode('.png', image)
        assert success, "Failed in 'cv2.imencode()'."
        display(_Image(data=buffer.tobytes()))
        return
    try:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        print(f"Failed to show image in 'cv2.imshow()': {repr(e)}")
        try:
            import matplotlib.pyplot as plt
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(rgb)
            plt.axis("off")
            plt.show()
        except Exception as fallback_error:
            print(f"Failed to show image in 'matplotlib.pyplot.imshow()': {repr(fallback_error)}")
            raise RuntimeError("Failed to find available backend to show image.")

def save_image(image, path=None, suffix=None, datatype="png", is_rgb=False, logger=None):
    """
    Save an image to disk.

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The image to be saved. See `convert_image()` for compatibility.
        path (str | None, optional): Path to the folder in which the image is to be saved.
            The folder will be created if it doesn't exist.
            Pass None to use the packages 'data' folder. Defaults to None.
        suffix (str | None, optional):
            Appends a suffix to the filename (format: stamp_random_suffix).
            Pass None to not append a suffix. Defaults to None.
        datatype (str, optional):
            Datatype of the written file. Supported are: 'png' and 'jpeg'. Defaults to 'png'.
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (True) or BGR (False) order. Defaults to False.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to None.
    Raises:
        AssertionError: If arguments are invalid.
        RuntimeError: If operation unexpectedly fails.

    Returns:
        str: Path to the saved image file.

    TODOs:
        - Expose quality/compression settings like [cv2.IMWRITE_PNG_COMPRESSION, png_compression_level]
    """
    # parse arguments
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'", logger=None)
    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'image'", logger=logger)
    assert_type_value(obj=path, type_or_value=[str, None], name="argument 'path'", logger=logger)
    assert_type_value(obj=suffix, type_or_value=[str, None], name="argument 'suffix'", logger=logger)
    assert_type_value(obj=datatype, type_or_value=["png", "jpeg"], name="argument 'datatype'", logger=logger)
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'", logger=logger)

    # convert image
    image = convert_image(image, target_format="NumPy", target_encoding="bgr8", is_rgb=is_rgb)

    # set target folder
    if path is None:
        path = os.path.join(get_package_path("nimbro_utils"), "data", "save_image")
    if not os.path.exists(path):
        if logger is not None:
            logger.debug(f"Creating directory '{path}'")
        try:
            os.makedirs(path)
        except Exception as e:
            message = f"Failed to create directory '{path}': {repr(e)}"
            if logger is not None:
                logger.error(message)
            raise RuntimeError(message)
    elif not os.path.isdir(path):
        message = f"Expected path '{path}' to either not exist or be a directory."
        if logger is not None:
            logger.error(message)
        assert os.path.isdir(path), message

    # set file name
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"
    stamp = convert_stamp(stamp=time.time(), target_format="iso")[:-7].replace("-", "_").replace(":", "_")
    random = uuid.uuid4().hex[:8]
    file_name = f"{stamp}_{random}{suffix}.{datatype}"
    image_path = os.path.join(path, file_name)

    # save image
    if logger is not None:
        logger.debug(f"Saving image '{image_path}'")
    try:
        cv2.imwrite(image_path, image)
    except Exception as e:
        message = f"Failed to save image '{image_path}': {repr(e)}"
        if logger is not None:
            logger.error(message)
        raise RuntimeError(message)

    return image_path

def convert_image(image, target_format="NumPy", target_encoding=None, is_rgb=False, is_depth=False, logger=None):
    """
    Convert an input image of various types into a specified output format and encoding.

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The source image to convert. Supported formats:
              - str: A URL (http/https) or local file path.
              - numpy.ndarray: A NumPy ndarray with dtype 'numpy.uint8' or 'numpy.uint16', and 1 or 3 channels.
              - sensor_msgs.msg.Image: A ROS2 Image message.
              - sensor_msgs.msg.CompressedImage: A ROS2 CompressedImage message.
        target_format (str, optional):
            Desired output container. One of:
              - 'NumPy' to return a numpy.ndarray.
              - 'Image' to return a sensor_msgs.msg.Image.
              - 'Compressed' to return a sensor_msgs.msg.CompressedImage.
            Defaults to 'NumPy'.
        target_encoding (str | None, optional):
            Use str to determine the channel count, bit depth, and color order. One of:
              '16UC1', 'mono8', 'mono16', 'bgr8', 'rgb8', 'bgr16', 'rgb16'.
            Use None to leave the encoding as is. Defaults to None.
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (True) or BGR (False) order. Defaults to False.
        is_depth (bool, optional):
            For 1-channel NumPy inputs with dtype 'numpy.uint16', indicates if the data is depth (True) or mono color (False). Defaults to False.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to None.
    Raises:
        AssertionError: If arguments are invalid.
        RuntimeError: If operation unexpectedly fails.

    Returns:
        np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage:
            The converted image in the requested format and encoding.

    TODOs:
        - Accept b64 encoded images
        - Convert to b64 encoded images
    """
    # parse arguments
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'", logger=None)
    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'image'", logger=logger)
    assert_type_value(obj=target_format, type_or_value=str, name="argument 'target_format'", logger=logger)
    target_format = target_format.lower()
    assert_type_value(obj=target_format, type_or_value=["numpy", "image", "compressed"], name="argument 'target_format'", logger=logger)
    assert_type_value(obj=target_encoding, type_or_value=list(IMAGE_ENCODINGS.keys()) + [None], name="argument 'target_encoding'", logger=logger)
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'", logger=logger)
    assert_type_value(obj=is_depth, type_or_value=bool, name="argument 'is_depth'", logger=logger)

    if target_encoding is not None:
        dst_dtype, dst_channels, dst_color = IMAGE_ENCODINGS[target_encoding]

    # convert input to ndarray

    if isinstance(image, str):
        if is_url(image):
            # download
            success, message, image = download_image(url=image, rgb=False, logger=logger)
            assert success, message
            if image.ndim == 3:
                is_rgb = False
            if not success:
                raise RuntimeError(message)
        else:
            # read file
            if not os.path.isfile(image):
                message = f"Expected '{image}' to be a valid file path."
                if logger is not None:
                    logger.error(message)
                assert os.path.isfile(image), message
            if logger is not None:
                logger.debug(f"Loading image '{image}'")
            image = cv2.imread(image, cv2.IMREAD_COLOR)
            if image is None:
                message = f"Failed to load image from '{image}'."
                if logger is not None:
                    logger.error(message)
                raise RuntimeError(message)
            if image.ndim == 3:
                is_rgb = False

    if isinstance(image, np.ndarray):
        # validate shape and dtype
        if image.dtype not in [np.uint8, np.uint16]:
            message = f"Expected type of NumPy image to be in {[np.uint8, np.uint16]} but got '{image.dtype}'."
            if logger is not None:
                logger.error(message)
            assert image.dtype in [np.uint8, np.uint16], message
        if image.ndim == 2:
            pass
        elif image.ndim == 3:
            if image.shape[2] != 3:
                message = f"Expected NumPy image to have one or three channels but got shape '{image.shape}'."
                if logger is not None:
                    logger.error(message)
                assert image.shape[2] == 3, message
        else:
            message = f"Expected NumPy image to have one or three channels but got shape '{image.shape}'."
            if logger is not None:
                logger.error(message)
            assert image.ndim in [2, 3], message

        # set color
        if image.ndim == 3:
            current_color = "RGB" if is_rgb else "BGR"
        else:
            current_color = "DEPTH" if is_depth and image.dtype == np.uint16 else "GRAY"

        # keep encoding
        if target_encoding is None:
            if image.ndim == 3:
                if image.dtype == np.uint8:
                    target_encoding = "rgb8" if is_rgb else "bgr8"
                else:
                    target_encoding = "rgb16" if is_rgb else "bgr16"
            else:
                if image.dtype == np.uint8:
                    target_encoding = "mono8"
                elif image.dtype == np.uint16:
                    target_encoding = "16UC1" if is_depth else "mono16"
            dst_dtype, dst_channels, dst_color = IMAGE_ENCODINGS[target_encoding]

    elif isinstance(image, Image):
        # fast forward
        if target_format == "image" and target_encoding is None:
            if logger is not None:
                logger.debug(f"Fast forwarding format '{target_format}' without conversion")
            return image

        # validate message encoding
        if image.encoding not in IMAGE_ENCODINGS:
            message = f"Expected encoding of Image message to be in {list(IMAGE_ENCODINGS.keys())} but got '{image.encoding}'."
            if logger is not None:
                logger.error(message)
            assert image.encoding in IMAGE_ENCODINGS, message
        current_color = IMAGE_ENCODINGS[image.encoding][2]

        # fast forward
        if target_format == "image" and image.encoding == target_encoding:
            if logger is not None:
                logger.debug(f"Fast forwarding format '{target_format}' and encoding '{target_encoding}' without conversion")
            return image

        # keep encoding
        if target_encoding is None:
            target_encoding = image.encoding
            dst_dtype, dst_channels, dst_color = IMAGE_ENCODINGS[target_encoding]

        # convert to ndarray
        if logger is not None:
            logger.debug("Converting Image message to NumPy")
        image = bridge.imgmsg_to_cv2(image, desired_encoding="passthrough")
        if image is None:
            message = f"CvBridge returned None for Image message with encoding '{image.encoding}'."
            if logger is not None:
                logger.error(message)
            raise RuntimeError(message)

    elif isinstance(image, CompressedImage):
        # fast forward
        if target_format == "compressed" and target_encoding is None:
            if logger is not None:
                logger.debug(f"Fast forwarding format '{target_format}' without conversion")
            return image

        # parse image.format
        valid = False
        is_depth_transport = False
        parts = image.format.split(';')
        if len(parts) == 2:
            parts[0] = parts[0].strip()
            parts[1] = parts[1].strip()
            if parts[1].startswith("compressedDepth"):
                codec_part = parts[1].replace("compressedDepth", "").strip()
                if codec_part == "png":
                    valid = True
                    is_depth_transport = True
                    comp_pixfmt = "16UC1"
                else:
                    message = f"Expected codec of CompressedImage message with depth transport to be 'png' but got '{codec_part}'."
                    if logger is not None:
                        logger.error(message)
                    assert codec_part == "png", message
            elif "compressed" in parts[1]:
                comp_pixfmt = None
                sub_parts = parts[1].split('compressed')
                if len(sub_parts) == 2 and "compressedDepth" not in parts[1]:
                    sub_parts[0] = sub_parts[0].strip()
                    sub_parts[1] = sub_parts[1].strip()
                    if sub_parts[0] == "jpeg":
                        if sub_parts[1] in ["", "bgr8", "rgb8", "mono8"]:
                            valid = True
                            if sub_parts[1] != "":
                                comp_pixfmt = sub_parts[1]
                        else:
                            message = f"Expected encoding of CompressedImage message with color transport to be in ['', 'bgr8', 'rgb8', 'mono8'] for codec 'jpeg' but got '{sub_parts[1]}'."
                            if logger is not None:
                                logger.error(message)
                            assert sub_parts[1] in ["", "bgr8", "rgb8", "mono8"], message
                    elif sub_parts[0] == "png":
                        if sub_parts[1] in ["", "bgr8", "rgb8", "bgr16", "rgb16", "mono8", "mono16"]:
                            valid = True
                            if sub_parts[1] != "":
                                comp_pixfmt = sub_parts[1]
                        else:
                            message = f"Expected encoding of CompressedImage message with color transport to be in ['', 'bgr8', 'rgb8', 'mono8', 'bgr16', 'rgb16', 'mono16'] for codec 'png' but got '{sub_parts[1]}'."
                            if logger is not None:
                                logger.error(message)
                            assert sub_parts[1] in ["", "bgr8", "rgb8", "bgr16", "rgb16", "mono8", "mono16"], message
                    else:
                        message = f"Expected codec of CompressedImage message with color transport to be in ['jpeg', 'png'] but got '{sub_parts[1]}'."
                        if logger is not None:
                            logger.error(message)
                        assert sub_parts[0] in ["jpeg", "png"], message
        if not valid:
            message = f"Expected format of CompressedImage message to be 'ORIG_PIXFMT; CODEC compressed [COMPRESSED_PIXFMT]' or 'ORIG_PIXFMT; compressedDepth CODEC' but got '{image.format}'."
            if logger is not None:
                logger.error(message)
            assert valid, message

        # fast forward
        if target_format == "compressed" and comp_pixfmt == target_encoding:
            if logger is not None:
                logger.debug(f"Fast forwarding format '{target_format}' and encoding '{target_encoding}' without conversion")
            return image

        # fix data not starting with PNG signature too make CvBridge work
        if is_depth_transport:
            PNG_SIG = b"\x89PNG\r\n\x1a\n"
            raw = bytes(image.data)
            idx = raw.find(PNG_SIG)
            if idx == -1:
                message = "Expected to find PNG signature in data of CompressedImage message."
                if logger is not None:
                    logger.error(message)
                assert idx > -1, message
            if idx > 0:
                if logger is not None:
                    logger.debug(f"Data of CompressedImage message with depth transport does not begin with PNG signature, which begins at data index '{idx}'")
                image.data = raw[idx:]

        # convert to ndarray
        if logger is not None:
            logger.debug("Converting CompressedImage message to NumPy")
        image = bridge.compressed_imgmsg_to_cv2(image, desired_encoding="passthrough")
        if image is None:
            message = f"CvBridge returned None for CompressedImage message with format '{image.format}'."
            if logger is not None:
                logger.error(message)
            raise RuntimeError(message)

        # assess color
        if is_depth_transport:
            current_color = "DEPTH"
        else:
            if comp_pixfmt is None:
                if image.ndim == 2:
                    if image.dtype == np.uint8:
                        comp_pixfmt = "mono8"
                    elif image.dtype == np.uint16:
                        comp_pixfmt = "mono16"
                    else:
                        message = f"Expected type of CompressedImage with color transport converted to NumPy to be 8-bit or 16-bit but got '{image.dtype}'."
                        if logger is not None:
                            logger.error(message)
                        assert image.dtype in [np.uint8, np.uint16], message
                else:
                    comp_pixfmt = "rgb8" if is_rgb else "bgr8"
                if logger is not None:
                    logger.debug(f"Falling back to encoding '{comp_pixfmt}' after incomplete format field of CompressedImage message with color transport '{image.format}'.")
            if comp_pixfmt.startswith("rgb"):
                current_color = "RGB"
            elif comp_pixfmt.startswith("bgr"):
                current_color = "BGR"
            else:
                current_color = "GRAY"

        # keep encoding
        if target_encoding is None:
            target_encoding = comp_pixfmt
            dst_dtype, dst_channels, dst_color = IMAGE_ENCODINGS[target_encoding]

    else:
        raise NotImplementedError(f"Unsupported 'image' type '{type(image).__name__}'")

    # validate state
    if image.dtype not in [np.uint8, np.uint16]:
        message = f"Expected type of NumPy image to be in {[np.uint8, np.uint16]} but got '{image.dtype}'."
        if logger is not None:
            logger.error(message)
        assert image.dtype in [np.uint8, np.uint16], message
    if image.ndim == 2:
        if current_color not in ["GRAY", "DEPTH"]:
            message = f"Expected obtained color of 1-channel NumPy image to be in {['GRAY', 'DEPTH']} but got '{current_color}'."
            if logger is not None:
                logger.error(message)
            raise RuntimeError(message)
    elif image.ndim == 3:
        if image.shape[2] != 3:
            message = f"Expected NumPy image to have one or three channels but got shape '{image.shape}'."
            if logger is not None:
                logger.error(message)
            assert image.shape[2] == 3, message
        if current_color not in ["BGR", "RGB"]:
            message = f"Expected obtained color of 3-channel NumPy image to be in {['BGR', 'RGB']} but got '{current_color}'."
            if logger is not None:
                logger.error(message)
            raise RuntimeError(message)
    else:
        message = f"Expected NumPy image to have one or three channels but got shape '{image.shape}'."
        if logger is not None:
            logger.error(message)
        assert image.ndim in [2, 3], message

    # fast forward
    if target_format == "numpy" and target_encoding is None:
        if logger is not None:
            logger.debug(f"Fast forwarding format '{target_format}' without conversion")
        return image

    # convert to target dtype and encoding

    log_post_conversion = False
    if logger is not None and ((dst_channels == 1 and image.ndim == 3) or (dst_channels == 3 and image.ndim == 2) or image.dtype != dst_dtype):
        log_post_conversion = True
        logger.debug(f"Converting image from {'1' if image.ndim == 2 else '3'}-channel {current_color} in {image.dtype} (value range '{np.min(image)}' to '{np.max(image)}') to {dst_channels}-channel {dst_color} in {dst_dtype.__name__}.")

    # channel count
    if dst_channels == 1 and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY if current_color == "RGB" else cv2.COLOR_BGR2GRAY)
    elif dst_channels == 3 and image.ndim == 2:
        image = cv2.merge([image] * 3)

    # color order
    elif current_color != dst_color and current_color in ["RGB", "BGR"] and dst_color in ["RGB", "BGR"]:
        code = cv2.COLOR_RGB2BGR if current_color == "RGB" else cv2.COLOR_BGR2RGB
        image = cv2.cvtColor(image, code)

    # scale and convert dtype
    if image.dtype != dst_dtype:
        if image.dtype == np.uint8 and dst_dtype == np.uint16:
            image = image.astype(np.uint16) * 257
        elif image.dtype == np.uint16 and dst_dtype == np.uint8:
            image = (image // 256).astype(np.uint8)
        else:
            raise NotImplementedError(f"Unsupported 'image' type conversion from '{image.dtype}' to '{dst_dtype}'.")

    if log_post_conversion:
        logger.debug(f"Value range after conversion within NumPy is '{np.min(image)}' to '{np.max(image)}'")

    # convert format
    if target_format == "numpy":
        result = image
    elif target_format == "image":
        if logger is not None:
            logger.debug(f"Converting NumPy with shape '{image.shape}' and type '{image.dtype}' to Image message with encoding '{target_encoding}'")
        result = bridge.cv2_to_imgmsg(image, encoding=target_encoding)
        if image is None:
            message = f"CvBridge returned None for NumPy image with shape '{image.shape}' and type '{image.dtype}'."
            if logger is not None:
                logger.error(message)
            raise RuntimeError(message)
    elif target_format == "compressed":
        codec = "png"
        if logger is not None:
            logger.debug(f"Converting NumPy with shape '{image.shape}' and type '{image.dtype}' to CompressedImage message with format '{codec}'")
        result = bridge.cv2_to_compressed_imgmsg(image, dst_format=codec)
        if image is None:
            message = f"CvBridge returned None for NumPy image with shape '{image.shape}' and type '{image.dtype}'."
            if logger is not None:
                logger.error(message)
            raise RuntimeError(message)
        if dst_color == "DEPTH":
            result.format = f"{target_encoding}; compressedDepth {codec}"
        else:
            result.format = f"{target_encoding}; {codec} compressed {target_encoding}"

    # log
    if logger is not None:
        logger.debug(image_info(result, name="Conversion result"))

    return result

def download_image(url, rgb=False, retry=1, logger=None):
    """
    Download an image from the internet.

    Args:
        url (str): The URL from which to download the image.
        rgb (bool, optional): If True, the image is returned in RGB order instead of BGR. Defaults to False.
        retry (bool | int, optional): If True, retry until successful. If int > 0, retry this often before returning a failure.
        logger (RcutilsLogger | nimbro_utils.node_extensions.logger.Logger | None, optional):
            If provided, logs status/error messages. Defaults to None.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        tuple[bool, str, numpy.ndarray]: Success flag, status message, image if successful.
    """
    # parse arguments
    stamp = time.perf_counter()
    assert_type_value(obj=logger, type_or_value=[RcutilsLogger, Logger, None], name="argument 'logger'", logger=None)
    assert_type_value(obj=url, type_or_value=str, name="argument 'url'", logger=logger)
    assert_type_value(obj=rgb, type_or_value=bool, name="argument 'rgb'", logger=logger)
    assert_type_value(obj=retry, type_or_value=[bool, int], name="argument 'retry'", logger=logger)

    # import
    try:
        import requests
    except ImportError:
        message = "Cannot download image because the module 'requests' is not available."
        if logger is not None:
            logger.error(message)
        return False, message, None

    # download

    message = None
    while True:
        if message is not None:
            if logger is not None:
                logger.warn(f"{message} Retrying...")
            message = None

        if logger is not None:
            logger.debug(f"Downloading image from '{url}'")

        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
        except Exception as e:
            message = f"Failed to download image: {repr(e)}"
            if isinstance(retry, bool) and retry is True:
                continue
            elif isinstance(retry, int) and retry > 0:
                retry -= 1
                continue
            if logger is not None:
                logger.error(message)
            return False, message, None
        try:
            img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR) # bgr
            if img is None:
                raise RuntimeError("Failed to decode downloaded image.")
            if len(img.shape) != 3:
                raise RuntimeError(f"Shape of downloaded image is '{img.shape}'.")
            if rgb is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # rgb
        except Exception as e:
            message = f"Failed to decode downloaded content as '{'RGB' if rgb is True else 'BGR'}' image: {repr(e)}"
            if isinstance(retry, bool) and retry is True:
                continue
            elif isinstance(retry, int) and retry > 0:
                retry -= 1
                continue
            if logger is not None:
                logger.error(message)
            return False, message, None

        break

    message = f"Successfully downloaded image from '{url}' with shape {img.shape} of type '{img.dtype}' in range {[np.min(img), np.max(img)]} as '{'RGB' if rgb is True else 'BGR'}' in '{time.perf_counter() - stamp:.3f}s'."
    if logger is not None:
        logger.debug(message)

    return True, message, img

def resize_image(image, value, mode="scale", is_rgb=False, is_depth=False):
    """
    Scales an image by a factor, to a target width, or to a target height.

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The image to be scaled. See `convert_image()` for compatibility.
        value (int | float | tuple | list):
            Resize the image based on `mode`.
            All numeric values must be greater than zero.
        mode (str, optional):
            Determines the scaling mode. Must be one of:
                - 'scale': Multiply both width and height by `value` (int | float).
                - 'area': Scale the image so that area matches `value` (int | float).
                    - Use int to scale both axis so that the image area becomes `value` pixels.
                    - Use float to scale both axis so that the image area is scaled by `value`.
                - 'height': Scale the image so that its height becomes `value` pixels (int | float).
                - 'width': Scale the image so that its width becomes `value` pixels (int | float).
                - 'fit': Scale both axes of the image equally so that it fits a frame passing a 2-tuple (height, width) to `value` (int | float).
                - 'stretch': Stretch the height and width of the image individually by passing a 2-tuple (height, width) to `value`.
                    - Use None to leave the axis as is.
                    - Use float to scale the axis.
                    - Use int to stretch the axis so that it becomes this many pixels.
            Defaults to 'scale'.
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (True) or BGR (False) order. Defaults to False.
        is_depth (bool, optional):
            For 1-channel NumPy inputs with dtype 'numpy.uint16', indicates if the data is depth (True) or mono color (False). Defaults to False.

    Raises:
        AssertionError: If arguments are invalid.

    Returns:
        np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage :
            The resized image in the same format and encoding (except str is returned as NumPy).
    """
    # parse arguments
    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'image'")
    assert_type_value(obj=mode, type_or_value=["scale", "area", "height", "width", "fit", "stretch"], name="argument 'mode'")
    if mode in ["fit", "stretch"]:
        assert_type_value(obj=value, type_or_value=[tuple, list], name=f"argument 'scale' with mode '{mode}'")
        assert len(value) == 2, f"Expected argument '{value}' with mode '{mode}' to contain '2' elements but got '{len(value)}'."
        if mode == "fit":
            for item in value:
                assert_type_value(obj=item, type_or_value=[int, float], name=f"all items in argument 'scale' with mode '{mode}'")
                assert item > 0, f"Expected all values in argument '{value}' with mode '{mode}' to be None or greater zero but got '{item}'."
        else:
            for item in value:
                assert_type_value(obj=item, type_or_value=[int, float, None], name=f"all items in argument 'scale' with mode '{mode}'")
                assert item is None or item > 0, f"Expected all values in argument '{value}' with mode '{mode}' to be None or greater zero but got '{item}'."
    else:
        assert_type_value(obj=value, type_or_value=[int, float], name=f"argument 'scale' with mode '{mode}'")
        assert value > 0, f"Expected value of argument '{value}' with mode '{mode}' to be greater zero but got '{value}'."
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")
    assert_type_value(obj=is_depth, type_or_value=bool, name="argument 'is_depth'")

    # assess
    if isinstance(image, Image):
        source_format = "Image"
        is_depth = True if image.encoding == "16UC1" else False
    elif isinstance(image, CompressedImage):
        source_format = "Compressed"
        is_depth = True if "compressedDepth" in image.format else False
    else:
        source_format = "NumPy"

    # convert
    image = convert_image(image, target_format="NumPy", target_encoding=None)

    # resize

    h, w = image.shape[:2]
    if mode == "scale":
        height = h * value
        width = w * value
    elif mode == "area":
        if isinstance(value, int):
            s = math.sqrt(value / (h * w))
        else:
            s = math.sqrt(value)
        height = h * s
        width = w * s
    elif mode == "height":
        height = value
        width = w * value / h
    elif mode == "width":
        height = h * value / w
        width = value
    elif mode == "fit":
        s = min(value[0] / h, value[1] / w)
        height = h * s
        width = w * s
    elif mode == "stretch":
        if isinstance(value[0], int):
            height = value[0]
        elif isinstance(value[0], float):
            height = h * value[0]
        else:
            height = h
        if isinstance(value[1], int):
            width = value[1]
        elif isinstance(value[1], float):
            width = w * value[1]
        else:
            width = w
    else:
        raise NotImplementedError(f"Unknown mode '{mode}'.")

    width = int(round(width))
    height = int(round(height))

    if image.ndim == 2:
        interpolation = cv2.INTER_NEAREST # depth and mono
    elif width * height > w * h:
        interpolation = cv2.INTER_CUBIC # upscale color
    else:
        interpolation = cv2.INTER_AREA # downscale color

    resized = cv2.resize(image, (width, height), interpolation=interpolation)

    # convert
    resized = convert_image(resized, target_format=source_format, target_encoding=None, is_rgb=is_rgb, is_depth=is_depth)

    return resized

def image_info(image, name="Image info"):
    """
    Returns image metadata formatted as a string.

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The image analyse. See `convert_image()` for compatibility.
        name (str, optional):
            Suffix of the returned string. Defeaults to 'Image info'.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        str: Image metadata.
    """
    assert_type_value(obj=name, type_or_value=str, name="argument 'name'")
    if isinstance(image, np.ndarray):
        info = {'type': type(image).__name__, 'shape': image.shape, 'dtype': str(image.dtype), 'min': np.min(image), 'max': np.max(image)}
    elif isinstance(image, Image):
        info = {'type': type(image).__name__, 'height': image.height, 'width': image.width, 'encoding': image.encoding, 'is_bigendian': image.is_bigendian, 'step': image.step, 'len(data)': len(image.data)}
    elif isinstance(image, CompressedImage):
        info = {'type': type(image).__name__, 'format': image.format, 'len(data)': len(image.data)}
    else:
        info = {'type': type(image).__name__}
    message = f"{name}: {info}"
    return message

# visualization

def visualize_depth(depth, image=None, image_is_rgb=False, **kwargs):
    """
    Visualize a depth image by mapping depth values to colors.

    Args:
        depth (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The depth image to be visualized. See `convert_image()` for compatibility.
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage | None, optional):
            An image to overlay with the colorized depth image. See `convert_image()` for compatibility. Defaults to None
        image_is_rgb (bool, optional):
            If `image` is a 3-channel NumPy array, indicates if the data is in RGB (True) or BGR (False) order. Defaults to False.

    Hidden args:
        min_range (int, float, optional):
            Minimum depth in meters to display (values below are clamped). Defaults to 0.2.
        max_range (int, float, optional):
            Maximum depth in meters to display (values above are clamped). Defaults to 5.0.
        colormap (str, optional):
            Name of the OpenCV colormap to use. Must be in ["turbo", "viridis", "plasma"]. Defaults to "turbo".
            See 'https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html'.
        invalid_to_white (bool, optional):
            If True, invalid depth pixels are mapped to white instead of black.
            Ignored if `image` is provided. Defaults to False.
        alpha (int, float, optional):
            Opacity of the overlayed `image` between 0.0 (only depth) and 1.0 (only image).
            Ignored if `image` is not provided. Defaults to 0.5.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        np.ndarray: A BGR uint8 image of the same resolution visualizing the depth image.
    """
    # parse arguments
    assert_type_value(obj=depth, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'depth'")
    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage, None], name="argument 'image'")
    assert_type_value(obj=image_is_rgb, type_or_value=bool, name="argument 'image_is_rgb'")
    min_range = kwargs.pop('min_range', 0.2)
    max_range = kwargs.pop('max_range', 5.0)
    colormap = kwargs.pop('colormap', "turbo")
    invalid_to_white = kwargs.pop('invalid_to_white', False)
    alpha = kwargs.pop('alpha', 0.5)
    assert len(kwargs) == 0, f"Unexpected keyword argument{'' if len(kwargs) == 1 else 's'} '{list(kwargs.keys())[0] if len(kwargs) == 1 else list(kwargs.keys())}'."
    assert_type_value(obj=min_range, type_or_value=[int, float], name="argument 'min_range'")
    assert 0.0 <= min_range, f"Expected value of argument 'alpha' to be greater or equal zero but got '{min_range}'."
    assert_type_value(obj=max_range, type_or_value=[int, float], name="argument 'max_range'")
    assert min_range < max_range, f"Expected value of argument 'max_range' to be greater than value of argument 'min_range' but got '{max_range}' and '{min_range}'."
    assert_type_value(obj=colormap, type_or_value=["turbo", "viridis", "plasma"], name="argument 'colormap'")
    assert_type_value(obj=invalid_to_white, type_or_value=bool, name="argument 'invalid_to_white'")
    assert_type_value(obj=alpha, type_or_value=[int, float], name="argument 'alpha'")
    assert 0.0 <= alpha <= 1.0, f"Expected value of argument 'alpha' to be in interval [0.0, 1.0] but got '{alpha}'."

    # prepare depth
    depth_np = convert_image(depth, target_format="NumPy", target_encoding="16UC1")
    minr = float(min_range) * 1000.0
    maxr = float(max_range) * 1000.0
    invalid = (depth_np == 0) | (depth_np == np.iinfo(np.uint16).max) | ~np.isfinite(depth_np)

    depth_np = np.clip(depth_np, minr, maxr)
    depth_np = (depth_np - minr) / (maxr - minr)
    depth_np = (1.0 - depth_np) * 255
    depth_np = np.clip(np.round(depth_np), 0, 255).astype(np.uint8)

    # apply color
    cmap_id = {"turbo": cv2.COLORMAP_TURBO, "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA}[colormap]
    depth_np = cv2.applyColorMap(depth_np, cmap_id)
    if image is None:
        depth_np[invalid] = 255 if invalid_to_white else 0
    else:
        image_np = convert_image(image, target_format="NumPy", target_encoding="bgr8", is_rgb=image_is_rgb)
        depth_np[invalid] = image_np[invalid]

    # overlay image
    if image is not None:
        _alpha = float(alpha)
        depth_np = cv2.addWeighted(image_np, 1 - _alpha, depth_np, _alpha, 0)

    return depth_np

def visualize_detections(image, boxes, masks=None, labels=None, box_format="xyxy_normalized", is_rgb=False, **kwargs):
    """
    Draws bounding boxes and labels on an image with customizable options.

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The image to be drawn on. See `convert_image()` for compatibility.
        boxes (list | tuple | numpy.ndarray):
            A list of bounding boxes. Each box must be a 4-element sequence according to `box_format`.
        masks (list | None. optional):
            A list of boolean masks as numpy.ndarray in the shape of `image` corresponding to `boxes`.
            Use None or list containing None to not show masks. Defaults to None.
        labels (list | None, optional):
            List of string labels for each bounding box. Use None to deactivate all or a single label. Defaults to None.
        box_format (str, optional):
            Format of the input boxes. See `nimbro_utils.utility.geometry.convert_boxes()`.
            One of ["xyxy_normalized", "xyxy_absolute", "xywh_normalized", "xywh_absolute"].
            Defaults to 'xyxy_normalized'
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (True) or BGR (False) order.
            Defaults to False.

    Hidden args:
        colors (str | list | tuple | dict, optional):
            Defines the main colors of the visualization.
                - Use 'auto' to automatically set a color per detection.
                - Use 'auto_class' to automatically select a color per unique detection label.
                - Pass a list of 3-tuples (tuple | list) defining an 8-bit BGR color per detection.
                - Pass a dict mapping detection labels (str | None) to colors as 3-tuples (tuple | list) of 8-bit BGR integers.
            Defaults to 'auto_class'.
        alpha (float, optional):
            Global opacity of the visualization in (0.0, 1.0]. Defaults to 0.9.
        box_thickness (int, optional):
            Thickness of the boxes in pixels. Defaults to 4.
        box_alpha (float, optional):
            Opacity of the boxes in [0.0, 1.0]. Defaults to 1.0.
        label_font_size (int, optional):
            Font size (>0) used for box labels. Defaults to 22.
        label_font_path (str, optional):
            Path to .ttf or .otf font file used.
            Defaults to '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'.
        label_padding (int, optional):
            Number of pixels of space around box labels. Defaults to 6.
        label_text_color (str | list | tuple | None, optional):
            Defines the colors of the label texts.
                - Use 'auto' to automatically select black or white per box based on contrast towards box color.
                - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all label texts.
                - Use None to adopt the color of the detection.
            Defaults to 'auto'.
        label_background_color (str | list | tuple | None, optional):
            Defines the colors of the label backgrounds.
                - Use 'auto' to adopt the color of the detection.
                - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all label backgrounds.
                - Use None to not disable background behind the the label texts.
            Defaults to 'auto'.
        label_text_alpha (float, optional):
            Opacity of the label texts in (0.0, 1.0]. Defaults to 1.0.
        label_background_alpha (float, optional):
            Opacity of the label background in [0.0, 1.0]. Defaults to 1.0.
        mask_color (str | list | tuple, optional):
            Defines the colors of the masks.
                - Use 'auto' to adopt the color of the detection.
                - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all masks.
                - Use None to adopt the color of the detection.
            Defaults to 'auto'.
        mask_alpha (float, optional):
            Opacity of the masks in [0.0, 1.0]. Defaults to 0.2.
        contour_thickness (int, optional):
            Thickness of mask contours in pixels. Defaults to 2.
        contour_color (str | list | tuple, optional):
            Defines the colors of the mask contours.
                - Use 'auto' to adopt the color of the detection.
                - Pass a 3-tuple (tuple | list) defining an 8-bit BGR color for all mask contours.
                - Use None to adopt the color of the detection.
            Defaults to 'auto'.
        contour_alpha (float, optional):
            Opacity of the contours around masks in [0.0, 1.0]. Defaults to 0.7.
        auto_color_palette (nimbro_utils.utility.color.ColorPalette, optional):
            Color palette used for automatic color selection. Defaults to nimbro.
        auto_color_shuffle (bool, optional):
            Shuffle automatic color selection. Defaults to False.
        draw_order (str, optional):
            Order in which detections are drawn in ['size', 'input'],
            where 'size' uses the bounding bo area in descending order
            and 'input' iterates `boxes` as is. Defaults to 'size'.
    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        np.ndarray: A BGR uint8 image of the same resolution visualizing the detections.

    TODOs:
        - fill box with box color based on fill_alpha
        - validate fonts and improve portability
        - replace all arguments in pixel units by relative units
    """
    # parse arguments

    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'image'")
    image = convert_image(image, target_format="NumPy", target_encoding="bgr8", is_rgb=is_rgb)
    assert_type_value(obj=box_format, type_or_value=["xyxy_normalized", "xyxy_absolute", "xywh_normalized", "xywh_absolute"], name="argument 'box_format'")
    assert_type_value(obj=boxes, type_or_value=[list, tuple, np.ndarray], name="argument 'boxes'")
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()
    if box_format in ["xyxy_normalized", "xywh_normalized"]:
        for box in boxes:
            assert_type_value(obj=box, type_or_value=[list, tuple], name="all values in argument 'boxes'")
            assert len(box) == 4, f"Expected all values in argument 'boxes' to contain '4' values but got '{len(box)}'."
            for value in box:
                assert_type_value(obj=value, type_or_value=float, name="all values in a single box in argument 'boxes'")
                assert value >= 0.0 and value <= 1.0, f"{value}"
    else:
        for box in boxes:
            assert_type_value(obj=box, type_or_value=[list, tuple], name="all values in argument 'boxes'")
            assert len(box) == 4, f"Expected all values in argument 'boxes' to contain '4' values but got '{len(box)}'."
        for value in box:
            assert_type_value(obj=value, type_or_value=int, name="all values in a single box in argument 'boxes'")
            assert value >= 0, f"{value}"
        assert box[0] > 0, f"Expected value in argument 'boxes' for x0 '{box[0]}' > 0."
        assert box[1] > 0, f"Expected value in argument 'boxes' for y0 '{box[1]}' > 0."
        assert box[2] > 0, f"Expected value in argument 'boxes' for x1/w '{box[2]}' > 0."
        assert box[3] > 0, f"Expected value in argument 'boxes' for y1/h '{box[3]}' > 0."
        if box_format == "xyxy_absolute":
            assert box[0] < box[2], f"Expected value in argument 'boxes' for x0 '{box[0]}' < x1 '{box[2]}'."
            assert box[1] < box[3], f"Expected value in argument 'boxes' for y0 '{box[1]}' < y1 '{box[3]}'."
            assert box[2] < image.shape[1], f"Expected value in argument 'boxes' for x1 '{box[2]}' < image width '{image.shape[1]}'."
            assert box[3] < image.shape[0], f"Expected value in argument 'boxes' for y1 '{box[3]}' < image height {image.shape[0]}'."
        else:
            assert box[0] + box[2] < image.shape[1], f"Expected value in argument 'boxes' for x0 '{box[0]}' + width '{box[2]}' < image width '{image.shape[1]}'."
            assert box[1] + box[3] < image.shape[0], f"Expected value in argument 'boxes' for y0 '{box[1]}' + height '{box[3]}' < image height '{image.shape[0]}'."
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")

    # read image
    if len(boxes) == 0:
        return image
    overlay = image.copy()

    assert_type_value(obj=masks, type_or_value=[list, tuple, None], name="argument 'masks'")
    if masks is not None:
        for i in range(len(masks)):
            assert_type_value(obj=masks[i], type_or_value=[list, tuple, np.ndarray, None], name="all values in argument 'masks'")
            if isinstance(masks[i], (list, tuple)):
                masks[i] = np.asarray(masks[i])
            if masks[i] is not None:
                assert masks[i].dtype == np.bool_, f"{masks[i].dtype}"
                assert masks[i].shape == image.shape[:2], f"{masks[i].shape} {image.shape[:2]}"
    assert_type_value(obj=labels, type_or_value=[list, tuple, None], name="argument 'labels'")
    if labels is not None:
        assert len(labels) == len(boxes), f"Expected the number of values in arguments 'boxes' and 'labels' to match but got '{len(boxes)}' and '{len(labels)}'."
        for label in labels:
            assert_type_value(obj=label, type_or_value=[str, None], name="all values in argument 'labels'")

    colors = kwargs.pop('colors', "auto_class")
    alpha = kwargs.pop('alpha', 0.9)
    box_thickness = kwargs.pop('box_thickness', 4)
    box_alpha = kwargs.pop('box_alpha', 1.0)
    label_font_size = kwargs.pop('label_font_size', 22)
    label_font_path = kwargs.pop('label_font_path', "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    label_padding = kwargs.pop('label_padding', 6)
    label_text_color = kwargs.pop('label_text_color', "auto")
    label_background_color = kwargs.pop('label_background_color', "auto")
    label_text_alpha = kwargs.pop('label_text_alpha', 1.0)
    label_background_alpha = kwargs.pop('label_background_alpha', 1.0)
    mask_color = kwargs.pop('mask_color', "auto")
    mask_alpha = kwargs.pop('mask_alpha', 0.2)
    contour_thickness = kwargs.pop('contour_thickness', 2)
    contour_color = kwargs.pop('contour_color', "auto")
    contour_alpha = kwargs.pop('contour_alpha', 0.7)
    auto_color_palette = kwargs.pop('auto_color_palette', nimbro if len(boxes) > 10 else nimbro.ten)
    auto_color_shuffle = kwargs.pop('auto_color_shuffle', False)
    draw_order = kwargs.pop('draw_order', "size")
    assert len(kwargs) == 0, f"Unexpected keyword argument{'' if len(kwargs) == 1 else 's'} '{list(kwargs.keys())[0] if len(kwargs) == 1 else list(kwargs.keys())}'."

    assert_type_value(obj=colors, type_or_value=["auto", "auto_class", list, tuple, None], name="argument 'colors'")
    if isinstance(colors, (list, tuple)):
        assert len(colors) == len(boxes), f"Expected the number of values in arguments 'boxes' and 'colors' to match but got '{len(boxes)}' and '{len(colors)}'."
        for color in colors:
            assert_type_value(obj=color, type_or_value=[list, tuple], name="all values in argument 'colors'")
            assert len(color) == 3, f"Expected all colors in argument 'colors' to contain '3' values but got '{len(color)}'."
            for value in color:
                assert_type_value(obj=value, type_or_value=int, name="all values in a color in argument 'colors'")
                assert 0 <= value <= 255, f"Expected all values in a color in argument 'colors' to be 8-bit values but got '{value}'."
    elif isinstance(colors, dict):
        if None not in colors:
            colors[None] = (0, 0, 0)
        for key in colors:
            assert_type_value(obj=key, type_or_value=[str, None], name="all keys in argument 'colors'")
            assert_type_value(obj=color[key], type_or_value=[list, tuple], name="all values in argument 'colors'")
            assert len(color[key]) == 3, f"Expected all colors in argument 'colors' to contain '3' values but got '{len(color)}'."
            for value in color[key]:
                assert_type_value(obj=value, type_or_value=int, name="all values in a color in argument 'colors'")
                assert 0 <= value <= 255, f"Expected all values in a color in argument 'colors' to be 8-bit values but got '{value}'."
    assert_type_value(obj=alpha, type_or_value=float, name="argument 'alpha'")
    assert alpha > 0, f"Expected value of argument 'alpha' to be greater zero but got '{alpha}'."
    assert alpha <= 1, f"Expected value of argument 'alpha' to be one or less but got '{alpha}'."
    assert_type_value(obj=box_thickness, type_or_value=int, name="argument 'box_thickness'")
    assert box_thickness > 0, "Expected value of argument 'box_thickness' to be greater zero."
    assert_type_value(obj=box_alpha, type_or_value=float, name="argument 'box_alpha'")
    assert box_alpha >= 0, f"Expected value of argument 'box_alpha' to be zero or greater but got '{box_alpha}'."
    assert box_alpha <= 1, f"Expected value of argument 'box_alpha' to be one or less but got '{box_alpha}'."
    assert_type_value(obj=label_font_size, type_or_value=int, name="argument 'label_font_size'")
    assert label_font_size > 0, f"Expected value of argument 'label_font_size' to be greater zero but got '{label_font_size}'."
    assert_type_value(obj=label_font_path, type_or_value=str, name="argument 'label_font_path'")
    assert_type_value(obj=label_padding, type_or_value=int, name="argument 'label_padding'")
    assert label_padding >= 0, f"Expected value of argument 'label_padding' to be zero or greater but got '{label_padding}'."
    assert_type_value(obj=label_text_color, type_or_value=["auto", list, tuple, None], name="argument 'label_text_color'")
    if isinstance(label_text_color, (list, tuple)):
        assert_type_value(obj=label_text_color, type_or_value=[list, tuple], name="all values in argument 'label_text_color'")
        assert len(label_text_color) == 3, f"Expected argument 'label_text_color' to contain '3' values but got '{len(label_text_color)}'."
        for value in label_text_color:
            assert_type_value(obj=value, type_or_value=int, name="all values in argument 'label_text_color'")
            assert 0 <= value <= 255, f"Expected all values in argument 'label_text_color' to be 8-bit values but got '{value}'."
    assert_type_value(obj=label_background_color, type_or_value=["auto", list, tuple, None], name="argument 'label_background_color'")
    if isinstance(label_background_color, (list, tuple)):
        assert_type_value(obj=label_background_color, type_or_value=[list, tuple], name="all values in argument 'label_background_color'")
        assert len(label_background_color) == 3, f"Expected argument 'label_background_color' to contain '3' values but got '{len(label_background_color)}'."
        for value in label_background_color:
            assert_type_value(obj=value, type_or_value=int, name="all values in argument 'label_background_color'")
            assert 0 <= value <= 255, f"Expected all values in argument 'label_background_color' to be 8-bit values but got '{value}'."
    assert_type_value(obj=label_text_alpha, type_or_value=float, name="argument 'label_text_alpha'")
    assert label_text_alpha >= 0, f"Expected value of argument 'label_text_alpha' to be zero or greater but got '{label_text_alpha}'."
    assert label_text_alpha <= 1, f"Expected value of argument 'label_text_alpha' to be one or less but got '{label_text_alpha}'."
    assert_type_value(obj=label_background_alpha, type_or_value=float, name="argument 'label_background_alpha'")
    assert label_background_alpha >= 0, f"Expected value of argument 'label_background_alpha' to be zero or greater but got '{label_background_alpha}'."
    assert label_background_alpha <= 1, f"Expected value of argument 'label_background_alpha' to be one or less but got '{label_background_alpha}'."
    assert_type_value(obj=mask_color, type_or_value=["auto", list, tuple], name="argument 'mask_color'")
    if isinstance(mask_color, (list, tuple)):
        assert_type_value(obj=mask_color, type_or_value=[list, tuple], name="all values in argument 'mask_color'")
        assert len(mask_color) == 3, f"Expected argument 'mask_color' to contain '3' values but got '{len(mask_color)}'."
        for value in mask_color:
            assert_type_value(obj=value, type_or_value=int, name="all values in argument 'mask_color'")
            assert 0 <= value <= 255, f"Expected all values in argument 'mask_color' to be 8-bit values but got '{value}'."
    assert_type_value(obj=mask_alpha, type_or_value=float, name="argument 'mask_alpha'")
    assert mask_alpha >= 0, f"Expected value of argument 'mask_alpha' to be zero or greater but got '{mask_alpha}'."
    assert mask_alpha <= 1, f"Expected value of argument 'mask_alpha' to be one or less but got '{mask_alpha}'."
    assert_type_value(obj=contour_thickness, type_or_value=int, name="argument 'contour_thickness'")
    assert contour_thickness >= 0, "Expected value of argument 'contour_thickness' to be zero or greater."
    assert_type_value(obj=contour_color, type_or_value=["auto", list, tuple], name="argument 'contour_color'")
    if isinstance(contour_color, (list, tuple)):
        assert_type_value(obj=contour_color, type_or_value=[list, tuple], name="all values in argument 'contour_color'")
        assert len(contour_color) == 3, f"Expected argument 'contour_color' to contain '3' values but got '{len(contour_color)}'."
        for value in contour_color:
            assert_type_value(obj=value, type_or_value=int, name="all values in argument 'contour_color'")
            assert 0 <= value <= 255, f"Expected all values in argument 'contour_color' to be 8-bit values but got '{value}'."
    assert_type_value(obj=contour_alpha, type_or_value=float, name="argument 'contour_alpha'")
    assert contour_alpha >= 0, f"Expected value of argument 'contour_alpha' to be zero or greater but got '{contour_alpha}'."
    assert contour_alpha <= 1, f"Expected value of argument 'contour_alpha' to be one or less but got '{contour_alpha}'."
    assert_type_value(obj=auto_color_palette, type_or_value=ColorPalette, name="argument 'auto_color_palette'")
    assert_type_value(obj=auto_color_shuffle, type_or_value=bool, name="argument 'auto_color_shuffle'")
    assert_type_value(obj=draw_order, type_or_value=["size", "input"], name="argument 'draw_order'")

    # determine colors
    if colors == "auto" or labels is None or all(label is None for label in labels):
        colors = []
        if auto_color_shuffle is True:
            while len(colors) < len(boxes):
                colors += auto_color_palette.bgr_shuffle[:len(boxes)]
        else:
            while len(colors) < len(boxes):
                colors += auto_color_palette.bgr[:len(boxes)]
    elif colors == "auto_class":
        unique_labels = []
        for label in labels:
            if label is not None and label not in unique_labels:
                unique_labels.append(label)
        assert len(unique_labels) > 0
        colors = []
        if auto_color_shuffle is True:
            while len(colors) < len(boxes):
                colors += auto_color_palette.bgr_shuffle[:len(boxes)]
        else:
            while len(colors) < len(boxes):
                colors += auto_color_palette.bgr[:len(boxes)]
        colors = dict(zip(unique_labels, colors))
    if isinstance(colors, dict):
        colors_per_box = []
        for i in range(len(boxes)):
            if labels is None:
                colors_per_box.append(colors[None])
            else:
                colors_per_box.append(colors[labels[i]])
        colors = colors_per_box

    # draw boxes and labels

    boxes = convert_boxes(boxes, source_format=box_format, target_format="xyxy_absolute", image_size=image.shape[:2])

    if draw_order == "size":
        sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        order = np.argsort(-np.array(sizes))
    elif draw_order == "input":
        order = range(len(boxes))

    for i in order:
        if masks is not None and masks[i] is not None:
            if mask_alpha > 0:
                overlay[masks[i]] = (1 - mask_alpha) * image[masks[i]] + mask_alpha * (np.array(colors[i]) if mask_color == "auto" else np.array(mask_color))
            if contour_thickness > 0:
                contours, _ = cv2.findContours(
                    image=(masks[i] * 255).astype(np.uint8),
                    mode=cv2.RETR_LIST, # RETR_LIST RETR_EXTERNAL
                    method=cv2.CHAIN_APPROX_NONE # CHAIN_APPROX_NONE CHAIN_APPROX_SIMPLE
                )

            h, w = masks[i].shape[:2]
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            color = colors[i] if contour_color == "auto" else contour_color

            for contour in contours:
                points = contour[:, 0]
                lines_to_draw = []
                for j in range(len(points)):
                    pt1 = tuple(points[j])
                    pt2 = tuple(points[(j + 1) % len(points)])
                    if any((
                        pt1[0] <= 0, pt1[0] >= w - 1, pt1[1] <= 0, pt1[1] >= h - 1,
                        pt2[0] <= 0, pt2[0] >= w - 1, pt2[1] <= 0, pt2[1] >= h - 1
                    )):
                        continue
                    lines_to_draw.append((pt1, pt2))

                for pt1, pt2 in lines_to_draw:
                    cv2.line(contour_mask, pt1, pt2, color=255, thickness=contour_thickness, lineType=cv2.LINE_AA)

            contour_mask = contour_mask.astype(bool)
            overlay[contour_mask] = (1 - contour_alpha) * overlay[contour_mask] + contour_alpha * np.array(color)

        if box_alpha > 0:
            overlay = draw_rectangle(image=overlay, box=boxes[i], box_format="xyxy_absolute", color=colors[i], thickness=box_thickness, alpha=box_alpha, is_rgb=False)

        if labels is not None and labels[i] is not None:
            display_text = labels[i]

            if label_text_color is None:
                text_color = colors[i]
            elif label_text_color == "auto":
                # calculate luminance using ITU-R BT.709 coefficients
                b, g, r = label_background_color if isinstance(label_background_color, (list, tuple)) else colors[i]
                luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
                if luminance > 0.5:
                    text_color = (0, 0, 0)
                else:
                    text_color = (255, 255, 255)
            else:
                text_color = label_text_color

            if label_background_color is None:
                background_color = None
            elif label_background_color == "auto":
                background_color = colors[i]
            else:
                background_color = label_background_color

            overlay = draw_text(
                image=overlay,
                text=display_text,
                anchor=(boxes[i][0] - box_thickness, boxes[i][1] - 1),
                font_path=label_font_path,
                font_size=label_font_size,
                text_color=text_color,
                background_color=background_color,
                padding=label_padding,
                line_gap=label_padding,
                is_rgb=False,
                background_alpha=label_background_alpha,
                text_alpha=label_text_alpha,
            )

    # blend the overlay with the original image
    image = np.round(cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)).astype(np.uint8)

    return image

def draw_rectangle(image, box, box_format="xyxy_normalized", is_rgb=False, **kwargs):
    """
    Draw an unfilled rectangle on an image.

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The image to be drawn on. See `convert_image()` for compatibility.
        box (tuple):
            A bounding box as a 4-element sequence according to `box_format`.
            A bounding box. See `nimbro_utils.utility.geometry.convert_boxes()`.
        box_format (str, optional):
            Format of the input boxes. See `nimbro_utils.utility.geometry.convert_boxes()`.
            One of ["xyxy_normalized", "xyxy_absolute", "xywh_normalized", "xywh_absolute"].
            Defaults to 'xyxy_normalized'
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (True) or BGR (False) order.
            Defaults to False.

    Hidden args:
        color (tuple | list, optional):
            Color of the rectangle as 3-tuple (tuple | list) 8-bit BGR.
            Defaults to (255, 255, 255).
        thickness (int, optional):
            Extension of the rectangle in pixels (>0) outward from `box`.
            Defaults to 1.
        alpha (float, optional):
            Opacity of the rectangle in [0.0, 1.0].
            Defaults to 0.1.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        np.ndarray: A BGR uint8 image of the same resolution with the rectangle drawn on it.
    """
    # parse arguments
    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'image'")
    valid_formats = ["xyxy_absolute", "xyxy_normalized", "xywh_absolute", "xywh_normalized"]
    assert_type_value(obj=box_format, type_or_value=valid_formats, name="argument 'box_format'")
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")
    color = kwargs.pop('color', (255, 255, 255))
    thickness = kwargs.pop('thickness', 1)
    alpha = kwargs.pop('alpha', 1.0)
    assert len(kwargs) == 0, f"Unexpected keyword argument{'' if len(kwargs) == 1 else 's'} '{list(kwargs.keys())[0] if len(kwargs) == 1 else list(kwargs.keys())}'."
    assert_type_value(obj=color, type_or_value=[list, tuple], name="all values in argument 'colors'")
    assert len(color) == 3, f"Expected all colors in argument 'colors' to contain '3' values but got '{len(color)}'."
    for value in color:
        assert_type_value(obj=value, type_or_value=int, name="all values in a color in argument 'colors'")
        assert 0 <= value <= 255, f"Expected all values in a color in argument 'colors' to be 8-bit values but got '{value}'."
    assert_type_value(obj=thickness, type_or_value=int, name="argument 'thickness'")
    assert thickness > 0, f"Expected value of argument 'thickness' to be greater zero but got '{thickness}'."
    assert_type_value(obj=alpha, type_or_value=float, name="argument 'alpha'")
    assert alpha >= 0, f"Expected value of argument 'alpha' to be zero or greater but got '{alpha}'."
    assert alpha <= 1, f"Expected value of argument 'alpha' to be one or less but got '{alpha}'."

    # read image
    image = convert_image(image, target_format="NumPy", target_encoding="bgr8")
    if alpha == 0:
        return image
    h, w = image.shape[:2]

    # convert box
    x1, y1, x2, y2 = convert_boxes([box], source_format=box_format, target_format="xyxy_absolute", image_size=image.shape[:2])[0]

    # draw lines outward up to thickness
    for t in range(thickness):
        # horizontal lines (top and bottom)
        yt_top = y1 - (t + 1)
        yt_bottom = y2 + (t + 1)
        if 0 <= yt_top < h:
            x_start = max(0, x1 - (t + 1))
            x_end = min(w, x2 + (t + 1) + 1)
            image[yt_top, x_start:x_end] = (1 - alpha) * image[yt_top, x_start:x_end] + alpha * np.array(color)

        if 0 <= yt_bottom < h:
            x_start = max(0, x1 - (t + 1))
            x_end = min(w, x2 + (t + 1) + 1)
            image[yt_bottom, x_start:x_end] = (1 - alpha) * image[yt_bottom, x_start:x_end] + alpha * np.array(color)

        # vertical lines (left and right)
        xt_left = x1 - (t + 1)
        xt_right = x2 + (t + 1)
        if 0 <= xt_left < w:
            y_start = max(0, y1 - (t + 1))
            y_end = min(h, y2 + (t + 1) + 1)
            image[y_start:y_end, xt_left] = (1 - alpha) * image[y_start:y_end, xt_left] + alpha * np.array(color)
        if 0 <= xt_right < w:
            y_start = max(0, y1 - (t + 1))
            y_end = min(h, y2 + (t + 1) + 1)
            image[y_start:y_end, xt_right] = (1 - alpha) * image[y_start:y_end, xt_right] + alpha * np.array(color)

    return image

def draw_text(image, text, anchor=(0, 0), is_rgb=False, **kwargs):
    """
    Draw a text on an image.

    Args:
        image (str | np.ndarray | sensor_msgs.msg.Image | sensor_msgs.msg.CompressedImage):
            The image to be drawn on. See `convert_image()` for compatibility.
        text (str):
            Text to be drawn.
        anchor (tuple, list):
            Tuple of valid (x, y) pixel-coordinates defining the lower left corner of the drawn text.
            Defaults to (0, 0).
        is_rgb (bool, optional):
            For 3-channel NumPy inputs, indicates if the data is in RGB (True) or BGR (False) order.
            Defaults to False.

    Hidden args:
        font_path (str, optional):
            Path to .ttf or .otf font file used to draw text.
            Commonly found monospaced fonts are '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
            and '/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf'.
            Defaults to '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'.
        font_size (int, optional):
            Font size (>0) of the text. Defaults to 22.
        text_color (tuple | list, optional):
            Color of the text as 3-tuple (tuple | list) 8-bit BGR.
            Defaults to (255, 255, 255).
        background_color (tuple | list | None, optional):
            Color of the background behind the the text as 3-tuple (tuple | list) 8-bit BGR.
            Use None to not draw the background. Defaults to None.
        padding (int, optional):
            Space between text and background on all sides in pixels (>=0).
            Defaults to 4.
        line_gap (int, optional):
            Space between lines when text is a multiline-string in pixels (>=0).
            Defaults to 2.
        text_alpha (float, optional):
            Opacity of the text in [0.0, 1.0].
            Defaults to 1.0.
        background_alpha (float, optional):
            Opacity of the background in [0.0, 1.0].
            Defaults to 1.0.

    Raises:
        AssertionError: If input arguments are invalid.
        ImportError: If `pillow` is not available.

    Returns:
        np.ndarray: A BGR uint8 image of the same resolution with the text drawn on it.
    """
    # parse arguments
    assert_type_value(obj=image, type_or_value=[str, np.ndarray, Image, CompressedImage], name="argument 'image'")
    assert_type_value(obj=text, type_or_value=str, name="argument 'text'")
    assert_type_value(anchor, [tuple, list], name="argument 'anchor'")
    assert len(anchor) == 2, f"Expected argument 'anchor' to be a 2-tuple (x, y) but got a tuple of length '{len(anchor)}'."
    assert isinstance(anchor[0], int) and isinstance(anchor[1], int), "Expected argument 'anchor' to be a 2-tuple (x, y) of integers."
    assert_type_value(obj=is_rgb, type_or_value=bool, name="argument 'is_rgb'")
    font_path = kwargs.pop('font_path', "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font_size = kwargs.pop('font_size', 22)
    text_color = kwargs.pop('text_color', (255, 255, 255))
    background_color = kwargs.pop('background_color', None)
    padding = kwargs.pop('padding', 4)
    line_gap = kwargs.pop('line_gap', 4)
    text_alpha = kwargs.pop('text_alpha', 1.0)
    background_alpha = kwargs.pop('background_alpha', 1.0)
    assert len(kwargs) == 0, f"Unexpected keyword argument{'' if len(kwargs) == 1 else 's'} '{list(kwargs.keys())[0] if len(kwargs) == 1 else list(kwargs.keys())}'."
    assert_type_value(obj=font_path, type_or_value=str, name="argument 'font_path'")
    assert_type_value(obj=font_size, type_or_value=int, name="argument 'font_size'")
    assert font_size > 0, f"Expected 'font_size' to be greater zero but got '{font_size}'."
    assert_type_value(obj=text_color, type_or_value=[list, tuple], name="all values in argument 'text_color'")
    assert len(text_color) == 3, f"Expected argument 'text_color' to contain '3' values but got '{len(text_color)}'."
    for value in text_color:
        assert_type_value(obj=value, type_or_value=int, name="all values in argument 'text_color'")
        assert 0 <= value <= 255, f"Expected all values in argument 'text_color' to be 8-bit values but got '{value}'."
    assert_type_value(obj=background_color, type_or_value=[list, tuple, None], name="all values in argument 'background_color'")
    if background_color is not None:
        assert len(background_color) == 3, f"Expected argument 'background_color' to contain '3' values but got '{len(background_color)}'."
        for value in background_color:
            assert_type_value(obj=value, type_or_value=int, name="all values in argument 'background_color'")
            assert 0 <= value <= 255, f"Expected all values in argument 'background_color' to be 8-bit values but got '{value}'."
    assert_type_value(obj=padding, type_or_value=int, name="argument 'padding'")
    assert padding >= 0, f"Expected 'padding' to be zero or greater but got '{padding}'."
    assert_type_value(obj=line_gap, type_or_value=int, name="argument 'line_gap'")
    assert line_gap >= 0, f"Expected 'line_gap' to be zero or greater but got '{line_gap}'."
    assert_type_value(obj=text_alpha, type_or_value=float, name="argument 'text_alpha'")
    assert 0.0 <= text_alpha <= 1.0, f"Expected 'text_alpha' to be in [0.0, 1.0] but got '{text_alpha}'."
    assert_type_value(obj=background_alpha, type_or_value=float, name="argument 'background_alpha'")
    assert 0.0 <= background_alpha <= 1.0, f"Expected 'background_alpha' to be in [0.0, 1.0] but got '{background_alpha}'."

    # import
    from PIL import Image as ImagePIL
    from PIL import ImageDraw, ImageFont

    # read image
    image = convert_image(image, target_format="NumPy", target_encoding="bgr8")
    overlay = ImagePIL.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(font_path, font_size)
    text = normalize_string(
        string=text,
        remove_underscores=False,
        remove_punctuation=False,
        remove_common_specials=False,
        reduce_whitespaces=False,
        lowercase=False
    )
    lines = text.split('\n')

    # measure text size
    bboxes = [font.getbbox(line) for line in lines]
    heights = [bbox[3] - bbox[1] for bbox in bboxes]
    max_width = max(bbox[2] - bbox[0] for bbox in bboxes)
    total_text_height = sum(heights)

    # define background box
    box_width = max_width + 2 * padding
    box_height = total_text_height + 2 * padding + (len(lines) - 1) * line_gap
    x_anchor, y_anchor = anchor
    if x_anchor < 0:
        x_anchor = 0
    elif x_anchor > image.shape[1] - 1 - box_width:
        x_anchor = image.shape[1] - 1 - box_width
    if y_anchor < box_height:
        y_anchor = box_height
    elif y_anchor > image.shape[0] - 1:
        y_anchor = image.shape[0] - 1
    x0 = x_anchor
    y0 = y_anchor - box_height
    x1 = x_anchor + box_width
    y1 = y_anchor

    # draw background
    if background_color is not None and background_alpha > 0.0:
        fill = tuple(background_color) + (int(255 * background_alpha),)
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    # draw text
    y_cursor = y0 + padding
    for i, line in enumerate(lines):
        bbox = bboxes[i]
        line_height = bbox[3] - bbox[1]
        y_text = y_cursor - bbox[1]
        x_text = x_anchor + padding - bbox[0]
        fill = tuple(text_color) + (int(255 * text_alpha),)
        draw.text((x_text, y_text), line, font=font, fill=fill)
        y_cursor += line_height + line_gap

    # composite over original image
    image = ImagePIL.fromarray(image).convert("RGBA")
    result = ImagePIL.alpha_composite(image, overlay).convert("RGB")
    return np.array(result)

# masks

def encode_mask(mask):
    """
    Encode a mask as a base64 string of a PNG image.

    Args:
        mask (numpy.ndarray): A mask with dtype 'np.bool_'.

    Raises:
        AssertionError: If input arguments are invalid.
        RuntimeError: If operation unexpectedly fails.

    Returns:
        str: Encoded mask.

    Notes:
        - If available, this function uses the faster 'pybase64' module; otherwise, it falls back to the standard 'base64' module.
    """
    # parse arguments
    assert_type_value(obj=mask, type_or_value=np.ndarray, name="argument 'mask'")

    # encode
    mask_uint8 = mask.astype(np.uint8) * 255
    success, mask_png = cv2.imencode('.png', mask_uint8)
    if not success:
        raise RuntimeError("Failed to encode mask as PNG.")
    if PYBASE64_AVAILABLE:
        return pybase64.b64encode(mask_png).decode('utf-8')
    else:
        return base64.b64encode(mask_png).decode('utf-8')

def decode_mask(mask):
    """
    Decode an encoded mask as NumPy.

    Args:
        mask (numpy.ndarray): A mask encoded as a base64 string of a PNG image.

    Raises:
        AssertionError: If input arguments are invalid.
        RuntimeError: If operation unexpectedly fails.

    Returns:
        numpy.ndarray: Decoded mask.

    Notes:
        - If available, this function uses the faster 'pybase64' module; otherwise, it falls back to the standard 'base64' module.
    """
    # parse arguments
    assert_type_value(obj=mask, type_or_value=str, name="argument 'mask'")

    # decode
    if PYBASE64_AVAILABLE:
        mask_bytes = pybase64.b64decode(mask)
    else:
        mask_bytes = base64.b64decode(mask)
    mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_np = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
    if mask_np is None:
        raise RuntimeError("Failed to decode PNG as base64")
    return mask_np > 0

def erode_or_dilate_mask(mask, dilate=False, kernel_size=3, iterations=3):
    """
    Erode a single-channel mask.

    Args:
        mask (numpy.ndarray): 2D numpy array (binary or grayscale).
        dilate (bool, optional): If True, dilates mask instead of eroding it. Defaults to False.
        kernel_size (int, optional): Odd positive integer, size of the square erosion/dilation kernel. Defaults to '3'.
        iterations (int, optional): Positive integer, number of times to apply erosion/dilation. Defaults to '3'.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        numpy.ndarray: Mask with same shape and dtype and morphology erosion/dilation applied.
    """
    # parse arguments
    assert_type_value(obj=mask, type_or_value=np.ndarray, name="argument 'mask'")
    assert mask.ndim == 2, f"Expected 'mask' to have two dimensions but got '{mask.ndim}'."
    assert mask.size > 0, "Expected 'mask' to not be empty."
    assert_type_value(obj=dilate, type_or_value=bool, name="argument 'dilate'")
    assert_type_value(obj=kernel_size, type_or_value=int, name="argument 'kernel_size'")
    assert kernel_size >= 1, f"Expected 'kernel_size' to be greater zero but got '{kernel_size}'."
    assert kernel_size % 2 == 1, f"Expected 'kernel_size' to be odd but got '{kernel_size}'."
    assert_type_value(obj=iterations, type_or_value=int, name="argument 'iterations'")
    assert iterations >= 1, f"Expected 'iterations' to be greater zero but got '{iterations}'."

    orig_dtype = mask.dtype

    # normalize to uint8
    if orig_dtype != np.uint8:
        mmin, mmax = mask.min(), mask.max()
        if mmin < 0 or mmax == mmin:
            temp = np.clip(mask, 0, 255).astype(np.uint8)
        else:
            temp = ((mask - mmin) / (mmax - mmin) * 255).astype(np.uint8)
    else:
        temp = mask

    # erode/dilate and cast back
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if dilate:
        result = cv2.dilate(temp, kernel, iterations=iterations).astype(orig_dtype)
    else:
        result = cv2.erode(temp, kernel, iterations=iterations).astype(orig_dtype)

    return result
