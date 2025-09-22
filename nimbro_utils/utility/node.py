#!/usr/bin/env python3

import os
import time
import threading
import traceback

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy._rclpy_pybind11 import InvalidHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from nimbro_utils.utility.misc import escape, assert_type_value

class SelfShutdown(Exception):
    """
    When raised inside a Node created by `start_and_spin_node()` it is shutdown gracefully without error and traceback.
    A shutdown notification is shown, including the optional message passed to this exception
    """
    pass

def start_and_spin_node(node_cls, *, args=None, domain_id=None, node_args=None, num_threads=50, blocking=True, os_shutdown=False):
    """
    Instantiate and spin a node with a `MultiThreadedExecutor`.

    Args:
        node_cls (type):
            The class of the node to be started (subclass of rclpy.node.Node).
        args (any, optional):
            Arguments passed to the context: context.init(args=...).
            Defaults to None.
        domain_id (int | None):
            Set the DOMAIN_ID of the context. Use None to set RCL_DEFAULT_DOMAIN_ID. Defaults to None.
        node_args (dict, optional):
            Arguments passed to `node_cls`: node_cls(**node_args).
            Defaults to None.
        num_threads (int | None, optional):
            Passed to `MultiThreadedExecutor`: MultiThreadedExecutor(num_threads=...).
            Use None to set the CPU count.
            Defaults to 50.
        blocking (bool, optional):
            If True, spins the node in this thread, blocks it, and cleans context on interrupt, shutdown, or exception.
            If False, spins the node in a thread and returns immediately,
                requiring cleanup via `stop_node()`.
            Defaults to True.
        os_shutdown (bool, optional):
            If True, calls `os._exit(0)` instead of `context.try_shutdown()`.
            Ignored when `blocking` is False. Defaults to True.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        None | tuple: If `blocking` is False, returns tuple (node, executor, context, thread) instead of None.

    Notes:
        - Raise `SelfShutdown` from inside a node to trigger a graceful shutdown without errors.
        - Use `os_shutdown` to shutdown nodes that tend to be unresponsive.

        - Example of a compatible node class:
            class MyNodeClass(rclpy.node.Node):
                def __init__(self, name="my_node_name", *, context=None, **kwargs):
                    super().__init__(name, context=context, **kwargs)

        - Example for starting a node using `ros2 run` or `ros2 launch`:
            def main(args=None):
                start_and_spin_node(MyNodeClass, args=args)

            if __name__ == '__main__':
                main()

        - Example for starting a node inside a `JupyterNotebook`:
            node, executor, context, thread = start_and_spin_node(MyNodeClass, blocking=False)
    """
    # parse arguments
    assert_type_value(obj=node_cls, type_or_value=type, name="argument 'node_cls'")
    assert_type_value(obj=node_args, type_or_value=[dict, None], name="argument 'node_args'")
    assert_type_value(obj=num_threads, type_or_value=[int, None], name="argument 'num_threads'")
    assert_type_value(obj=blocking, type_or_value=bool, name="argument 'blocking'")
    assert_type_value(obj=os_shutdown, type_or_value=bool, name="argument 'os_shutdown'")

    # Create an explicit shared context and initialize it
    context = rclpy.Context()
    context.init(args=args, domain_id=domain_id)
    if blocking:
        rclpy.signals.install_signal_handlers(rclpy.signals.SignalHandlerOptions.ALL)
    print(f"{escape['darkgreen']}>{escape['darkcyan']} Starting node '{node_cls.__name__}'{escape['end']}")

    # instantiate the node with the shared context
    try:
        node = node_cls(context=context, **(node_args or {}))
    except KeyboardInterrupt:
        print(f"{escape['darkyellow']}> {escape['darkcyan']}Interrupted node '{node_cls.__name__}'{escape['end']}")
        return
    except SelfShutdown as e:
        context.try_shutdown()
        context.destroy()
        print(f"{escape['darkred']}> {escape['darkcyan']}Node '{node_cls.__name__}' shutting down{'' if str(e) == '' else (': ' + str(e))}{escape['end']}")
        if blocking:
            return
        else:
            return node_cls, None, None, None
    except Exception as e:
        print(f"{escape['red']}> {escape['darkcyan']}Node '{node_cls.__name__}' raised exception: {repr(e)}{escape['end']}")
        context.try_shutdown()
        context.destroy()
        if blocking:
            trace = traceback.format_exc()
            print(f"{escape['red']}{trace}{escape['end']}")
            return
        else:
            raise e

    # create executor with same context and add node
    executor = MultiThreadedExecutor(context=context, num_threads=num_threads)
    added = executor.add_node(node)
    assert added, (
        f"Failed to add node with context {node.context!r} "
        f"to executor with context {executor.context!r}"
    )

    self_shutdown = False

    # define spin logic
    def _spin_and_handle():
        nonlocal self_shutdown
        try:
            while context.ok() and not executor._is_shutdown:
                try:
                    executor.spin()
                except KeyboardInterrupt:
                    print(f"{escape['darkyellow']}> {escape['darkcyan']}Interrupted node '{type(node).__name__}'{escape['end']}")
                    return
                except SelfShutdown as e:
                    print(f"{escape['darkred']}> {escape['darkcyan']}Node '{type(node).__name__}' shutting down{'' if str(e) == '' else (': ' + str(e))}{escape['end']}")
                    self_shutdown = True
                    return
                except Exception as e:
                    if isinstance(e, InvalidHandle):
                        raise
                    elif isinstance(e, ValueError) and 'generator already executing' in str(e):
                        pass
                    else:
                        trace = traceback.format_exc()
                        print(f"{escape['red']}> {escape['darkcyan']}Node '{type(node).__name__}' raised exception: {repr(e)}{escape['end']}")
                        print(f"{escape['red']}{trace}{escape['end']}")
                        return
        except InvalidHandle:
            return

    # run
    if blocking:
        try:
            _spin_and_handle()
        finally:
            time.sleep(1.0)
            executor.shutdown()
            node.destroy_node()
            if os_shutdown:
                print(f"{escape['magenta']}> {escape['darkcyan']}Killing node '{type(node).__name__}'{escape['end']}")
                os._exit(0)
            context.try_shutdown()
            context.destroy()
            if not self_shutdown:
                print(f"{escape['darkred']}> {escape['darkcyan']}Stopping node '{type(node).__name__}'{escape['end']}")
    else:
        thread = threading.Thread(target=_spin_and_handle)
        thread.start()
        return node, executor, context, thread

def stop_node(node, executor, context, thread):
    """
    Stop a node using the environment returned by `start_and_spin_node()` executed with blocking set to False.

    Notes:
        - Example for starting and stopping a node inside a `JupyterNotebook`:
            <cell A>
                from nimbro_utils.lazy import start_and_spin_node, stop_node
                node_env = start_and_spin_node(MyNodeClass, blocking=False)
            <cell B>
                stop_node(*node_env)
    """
    if all(arg is None for arg in [executor, context, thread]):
        print(f"{escape['red']}> {escape['darkcyan']}Node '{node.__name__}' has already stopped{escape['end']}")
    else:
        executor.shutdown()
        node.destroy_node()
        context.try_shutdown()
        thread.join(5.0)
        if thread.is_alive():
            print(f"{escape['magenta']}> {escape['darkcyan']}Destroying context of unresponsive node '{type(node).__name__}'{escape['end']}")
        context.destroy()
        time.sleep(1.0)
        print(f"{escape['darkred']}> {escape['darkcyan']}Stopped node '{type(node).__name__}'{escape['end']}")

def block_until_future_complete(node, future, timeout=None):
    """
    Blocks the execution until the given future is complete or a timeout occurs.

    Args:
        node (Node): The node object which contains the executor to spin.
        future (Future): The future object to monitor for completion.
        timeout_sec (float | int | None, optional): The time in seconds to wait for the future to complete.
                                       Use None to block indefinitely. Defaults to None.

    Returns:
        bool: True if the future is completed successfully. False if the timeout expired before completion.

    Raises:
        AssertionError: If input arguments are invalid.

    Notes:
        - Example: Blocking until a service or action is done:
            success = block_until_future_complete(node, future, timeout_sec=5.0)
            if success: # alternatively use future.done()
                print("Future completed successfully")
            else:
                print("Future did not complete within the timeout period")
    """
    # parse arguments
    assert_type_value(obj=node, type_or_value=Node, name="argument 'node'")
    assert_type_value(obj=future, type_or_value=Future, name="argument 'future'")
    assert_type_value(obj=timeout, type_or_value=[float, int, None], name="argument 'timeout_sec'")

    condition = threading.Condition()
    done_flag = [False]

    def future_done_cb(_):
        with condition:
            done_flag[0] = True
            condition.notify_all()

    future.add_done_callback(future_done_cb)

    start_time = time.monotonic()

    if timeout is not None:
        timeout = float(timeout)

    with condition:
        while not done_flag[0]:
            time_left = None
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                time_left = max(0.0, timeout - elapsed)
                if time_left == 0.0:
                    return False

            try:
                node.executor.spin_once(timeout_sec=time_left)
            except KeyboardInterrupt:
                os._exit(0)
                # raise KeyboardInterrupt
            except Exception as e:
                if isinstance(e, ValueError) and 'generator already executing' in str(e):
                    pass # someone else is already spinning this executor
                else:
                    if not rclpy.ok():
                        os._exit(0)
                        # raise e
                    node.get_logger().error(f"{repr(e)}\n{traceback.format_exc()}")
                    if timeout is not None and (time.monotonic() - start_time) >= timeout:
                        return False

            condition.wait(timeout=0.01)

    return True

def create_throttled_subscription(node, msg_type, topic, callback, qos_profile, *, throttle=0.0, callback_group=None, event_callbacks=None, qos_overriding_options=None, raw=False):
    """
    Like `Node.create_subscription()`, but guarantees `callback` is called
    at most once every `throttle` seconds. Set throttle=0.0 to disable.
    """
    # if no throttling wanted, just forward
    if throttle <= 0.0:
        return node.create_subscription(
            msg_type,
            topic,
            callback,
            qos_profile,
            callback_group=callback_group,
            event_callbacks=event_callbacks,
            qos_overriding_options=qos_overriding_options,
            raw=raw,
        )

    # track last call time (in seconds)
    now = node.get_clock().now().nanoseconds * 1e-9
    last_called = [now - throttle]

    def _throttled(msg):
        t = node.get_clock().now().nanoseconds * 1e-9
        if t - last_called[0] >= throttle:
            last_called[0] = t
            callback(msg)

    return node.create_subscription(
        msg_type,
        topic,
        _throttled,
        qos_profile,
        callback_group=callback_group,
        event_callbacks=event_callbacks,
        qos_overriding_options=qos_overriding_options,
        raw=raw,
    )

def wait_for_message(node, topic_name, topic_type, qos=1, timeout=None):
    """
    Waits for a message on a specified ROS2 topic.

    This function creates a subscription to the specified topic, waits for a message to be received,
    and returns the result. It blocks until either a message is received or the timeout is reached.

    Args:
        node (Node): The node object that will create the subscription.
        topic_name (str): The name of the topic to subscribe to.
        topic_type (type): The message type of the topic (e.g., `std_msgs.msg.String`).
        qos (int | rclpy.qos.QoSProfile, optional): The Quality of Service (QoS) profile or integer level. Defaults to 1 (Best effort).
        timeout (float, optional): The timeout in seconds. If None, it will block indefinitely until a message is received.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        tuple[bool, Any]: Success flag, the message received or None after timeout.
    """
    # parse arguments
    assert_type_value(obj=node, type_or_value=Node, name="argument 'node'")
    assert_type_value(obj=topic_name, type_or_value=str, name="argument 'topic_name'")
    assert_type_value(obj=qos, type_or_value=[int, rclpy.qos.QoSProfile], name="argument 'qos'")
    assert_type_value(obj=timeout, type_or_value=[float, None], name="argument 'timeout'")

    future = Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)

    sub = node.create_subscription(
        msg_type=topic_type,
        topic=topic_name,
        callback=callback,
        qos_profile=qos,
        callback_group=MutuallyExclusiveCallbackGroup()
    )
    success = block_until_future_complete(
        node=node,
        future=future,
        timeout=timeout
    )
    node.destroy_subscription(sub)

    return success, future.result()
