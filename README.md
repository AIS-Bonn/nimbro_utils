# NimbRo Utilities

A diverse collection of ROS2 and robotics-related utilities designed for the [ROS2 Jazzy](https://docs.ros.org/en/jazzy/index.html) distribution.

## Overview

### Python Utilities

Features provided with all utilities include:

* Detailed docstrings throughout.
* [Jupyter Notebooks](./notebooks) with examples for easy prototyping.
* Extensive logging with options to attach and configure custom loggers.
* Assertions for checking the types and values of function arguments.
* Flexible input handling (e.g., images can be passed as path, URL, NumPy, or ROS2 message).
* Robust handling of failure cases (e.g. by employing timeouts, or returning success flags and logs).
* [Minimal dependencies](./requirements.txt) and fallbacks to standard library if faster modules are unavailable.

#### Categories

Utilities are organized into the following categories:

* **Color** ([.py](./nimbro_utils/utility/color.py)/[.ipynb](./notebooks/utility/color.ipynb)): Common color palettes and format conversions.
* **Geometry** ([.py](./nimbro_utils/utility/geometry.py)/[.ipynb](./notebooks/utility/geometry.ipynb)): Conversions between rotation formalisms and related ROS2 messages.
* **Image** ([.py](./nimbro_utils/utility/image.py)/[.ipynb](./notebooks/utility/image.ipynb)): Standard image handling (read, save, download, convert, resize) and visualization.
* **Misc.** ([.py](./nimbro_utils/utility/misc.py)/[.ipynb](./notebooks/utility/misc.ipynb)): General utilities including JSON I/O, assertions, and timestamp conversions.
* **Node** ([.py](./nimbro_utils/utility/node.py)/[.ipynb](./notebooks/utility/node.ipynb)): Tools for ROS2 node instantiation, spinning, context management, and executors.
* **String** ([.py](./nimbro_utils/utility/string.py)/[.ipynb](./notebooks/utility/string.ipynb)): String normalization, analysis, and JSON extraction.

### Node Extensions

Node extensions are classes that augment ROS2 nodes with specific functionalities, abstracting away underlying ROS2 interface complexities (managing and using interfaces, logging, etc.):

*   **DepthConverter** ([.py](./nimbro_utils/node_extensions/depth_converter.py)/[.ipynb](./notebooks/node_extensions/depth_converter.ipynb)): Projects depth images to pointclouds, supporting various camera models.
*   **Logger** ([.py](./nimbro_utils/node_extensions/logger.py)/[.ipynb](./notebooks/node_extensions/logger.ipynb)): Wraps `RcutilsLogger` with easier configuration and severity/prefix coupling.
*   **ParameterHandler** ([.py](./nimbro_utils/node_extensions/parameter_handler.py)/[.ipynb](./notebooks/node_extensions/parameter_handler.ipynb)): Handles node parameters, including declaration, validation, and updates.
*   **SensorInterface** ([.py](./nimbro_utils/node_extensions/sensor_interface.py)/[.ipynb](./notebooks/node_extensions/sensor_interface.ipynb)): Manages sensor subscriptions and allows synchronized data retrieval.
*   **TfOracle** ([.py](./nimbro_utils/node_extensions/tf_oracle.py)/[.ipynb](./notebooks/node_extensions/tf_oracle.ipynb)): Provides methods for `tf2` queries, transformations, and interpolations.

### Import Mechanisms

For convenience, a central import mechanism is provided:
* To import [all available utilities](./nimbro_utils/__init__.py):
  ```python
  from nimbro_utils import *
  ```
* To selectively import utilities and minimize dependencies:
  ```python
  from nimbro_utils.lazy import start_and_spin_node, stop_node
  ```

### Nodes

A few utility nodes are provided for testing purposes:

* **DepthImageToPointcloud** ([.py](./nimbro_utils/nodes/depth_image_to_pointcloud.py)): Publishes a pointcloud from a depth image and camera info.
* **TestImagePublisher** ([.py](./nimbro_utils/nodes/test_image_publisher.py)): Publishes a configurable test image.

To launch these node:
```bash
ros2 run nimbro_utils <node_name>
```

## Citation

If you utilize this package in your research, please cite one of our relevant publications.

* **Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking**<br>
    [[arXiv:2503.16538](https://arxiv.org/abs/2503.16538)]
    ```bibtex
    @article{paetzold25vlmgist,
        author={Bastian P{\"a}tzold and Jan Nogga and Sven Behnke},
        title={Leveraging Vision-Language Models for Open-Vocabulary Instance Segmentation and Tracking},
        journal={IEEE Robotics and Automation Letters (RA-L)},
        year={2025}
    }
    ```

* **A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics**<br>
    [[arXiv:2410.22997](https://arxiv.org/abs/2410.22997)]
    ```bibtex
    @article{bode24prompt,
        author={Jonas Bode and Bastian P{\"a}tzold and Raphael Memmesheimer and Sven Behnke},
        title={A Comparison of Prompt Engineering Techniques for Task Planning and Execution in Service Robotics},
        journal={International Conference on Humanoid Robots (Humanoids)},
        year={2024}
    }
    ```

* **RoboCup@Home 2024 OPL Winner NimbRo: Anthropomorphic Service Robots using Foundation Models for Perception and Planning**<br>
    [[arXiv:2412.14989](https://arxiv.org/abs/2412.14989)]
    ```bibtex
    @article{memmesheimer25robocup,
        author={Raphael Memmesheimer and Jan Nogga and Bastian P{\"a}tzold and Evgenii Kruzhkov and Simon Bultmann and Michael Schreiber and Jonas Bode and Bertan Karacora and Juhui Park and Alena Savinykh and Sven Behnke},
        title={{RoboCup@Home 2024 OPL Winner NimbRo}: Anthropomorphic Service Robots using Foundation Models for Perception and Planning},
        journal={RoboCup 2024: RoboCup World Cup XXVII},
        year={2025}
    }
    ```

## License

`nimbro_utils` is licensed under the BSD-3-Clause License.

## Author

Bastian Pätzold <paetzold@ais.uni-bonn.de>
