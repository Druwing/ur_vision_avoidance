"""Minimal stand-ins for rclpy, message packages and tf2_ros.

They let the node classes be imported and driven from plain unittest code, so
the real node logic (parameters, callbacks, JSON payloads, timers) is exercised
without a ROS installation. This is a smoke test of the node code, NOT a
replacement for running the nodes under ROS 2 + Gazebo.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

PARAM_OVERRIDES: dict[str, dict] = {}  # node name -> {param: value}


class FakeTime:
    def __init__(self, ns: int = 0):
        self.nanoseconds = ns

    def to_msg(self):
        return SimpleNamespace(sec=self.nanoseconds // 10**9, nanosec=self.nanoseconds % 10**9)


class FakeClock:
    def __init__(self):
        self.ns = 0

    def now(self):
        return FakeTime(self.ns)

    def set_seconds(self, seconds: float):
        self.ns = int(seconds * 1e9)


class Recorder:
    """Stand-in for a publisher; keeps every published message."""

    def __init__(self, topic):
        self.topic = topic
        self.messages = []

    def publish(self, message):
        self.messages.append(message)

    @property
    def last(self):
        return self.messages[-1] if self.messages else None


class FakeLogger:
    def __init__(self):
        self.records = []

    def _log(self, level):
        def log(message, *args, **kwargs):
            self.records.append((level, str(message)))

        return log

    def __getattr__(self, name):
        if name in ("info", "warn", "warning", "error", "debug"):
            return self._log(name)
        raise AttributeError(name)


class FakeNode:
    def __init__(self, name: str):
        self._name = name
        self._params = {}
        self._clock = FakeClock()
        self._logger = FakeLogger()
        self.publishers: dict[str, Recorder] = {}
        self.subscriptions: dict[str, object] = {}
        self.timers = []

    def declare_parameter(self, name, default=None):
        overrides = PARAM_OVERRIDES.get(self._name, {})
        self._params[name] = overrides.get(name, default)

    def get_parameter(self, name):
        return SimpleNamespace(value=self._params[name])

    def create_publisher(self, msg_type, topic, qos):
        recorder = Recorder(topic)
        self.publishers[topic] = recorder
        return recorder

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions[topic] = callback
        return callback

    def create_timer(self, period, callback):
        self.timers.append((period, callback))
        return callback

    def get_clock(self):
        return self._clock

    def get_logger(self):
        return self._logger

    def destroy_node(self):
        pass


class Msg:
    """Tiny message base: keyword init plus defaults."""

    _fields: dict = {}

    def __init__(self, **kwargs):
        for key, default in self._fields.items():
            value = kwargs.pop(key, None)
            setattr(self, key, default() if (value is None and callable(default)) else (default if value is None else value))
        if kwargs:
            raise TypeError(f"unexpected fields {sorted(kwargs)}")


def _msg(cls_name, /, **fields):
    return type(cls_name, (Msg,), {"_fields": fields})


def install() -> bool:
    """Install the fakes unless a real rclpy is importable. Returns True if faked."""
    try:
        import rclpy  # noqa: F401

        return False
    except ImportError:
        pass

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    rclpy = module("rclpy", init=lambda *a, **k: None, spin=lambda *a, **k: None, shutdown=lambda: None)
    rclpy.node = module("rclpy.node", Node=FakeNode)
    rclpy.qos = module("rclpy.qos", qos_profile_sensor_data=object())
    rclpy.duration = module("rclpy.duration", Duration=lambda seconds=0.0: SimpleNamespace(seconds=seconds))
    rclpy.time = module("rclpy.time", Time=lambda nanoseconds=0: FakeTime(nanoseconds))
    rclpy.executors = module(
        "rclpy.executors", ExternalShutdownException=type("ExternalShutdownException", (Exception,), {})
    )

    String = _msg("String", data="")
    module("std_msgs")
    module("std_msgs.msg", String=String, Empty=_msg("Empty"), Float32MultiArray=_msg("Float32MultiArray", data=list),
           Bool=_msg("Bool", data=False), Float32=_msg("Float32", data=0.0))
    module("builtin_interfaces")
    module("builtin_interfaces.msg", Duration=_msg("Duration", sec=0, nanosec=0))
    module("sensor_msgs")
    module("sensor_msgs.msg", JointState=_msg("JointState", name=list, position=list, velocity=list),
           CameraInfo=_msg("CameraInfo", k=lambda: [0.0] * 9), Image=_msg("Image"))
    module("trajectory_msgs")
    module("trajectory_msgs.msg", JointTrajectory=_msg("JointTrajectory", joint_names=list, points=list),
           JointTrajectoryPoint=_msg("JointTrajectoryPoint", positions=list, velocities=list, time_from_start=None))

    class Marker:
        SPHERE, ADD, DELETEALL = 2, 0, 3

        def __init__(self):
            self.header = SimpleNamespace(frame_id="", stamp=None)
            self.pose = SimpleNamespace(
                position=SimpleNamespace(x=0, y=0, z=0), orientation=SimpleNamespace(w=0)
            )
            self.scale = SimpleNamespace(x=0, y=0, z=0)
            self.color = SimpleNamespace(r=0, g=0, b=0, a=0)
            self.ns, self.id, self.type, self.action, self.lifetime = "", 0, 0, 0, None

    module("visualization_msgs")
    module("visualization_msgs.msg", Marker=Marker, MarkerArray=_msg("MarkerArray", markers=list))

    class TransformException(Exception):
        pass

    class Buffer:
        transform = None  # tests set this to SimpleNamespace(transform=...)
        calls = []

        def lookup_transform(self, target, source, time, timeout=None):
            Buffer.calls.append((target, source, time.nanoseconds))
            if Buffer.transform is None:
                raise TransformException("no transform")
            return Buffer.transform

    module("tf2_ros", Buffer=Buffer, TransformException=TransformException,
           TransformListener=lambda buffer, node: None)
    return True
