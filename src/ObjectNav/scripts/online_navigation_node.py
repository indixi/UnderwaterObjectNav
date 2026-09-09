#!/usr/bin/env python3
"""ROS-side online ObjectNav coordinator for system Python 3.8."""

import math
import threading
import time

import message_filters
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from objectnav.msg import NavigationStatus
from objectnav.srv import (
    CancelNavigation,
    CancelNavigationResponse,
    StartNavigation,
    StartNavigationResponse,
)
from objectnav_bc.online.debug_visualizer import DebugVisualizer
from objectnav_bc.online.inference_client import InferenceClient
from objectnav_bc.online.online_recorder import OnlineEpisodeRecorder
from robot_control.msg import ActionRequest, ActionStatus


SUPPORTED_GOALS = ("echinus",)


def rpy_from_quaternion(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) \
        if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    return roll, pitch, yaw


def ros_time_seconds(value):
    if value is None or value == rospy.Time():
        return 0.0
    return value.to_sec()


class OnlineNavigationNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.bridge = CvBridge()
        root = "~online/"
        self.vehicle_name = rospy.get_param(root + "vehicle_name", "bricsbot")
        self.scene_id = rospy.get_param(root + "scene_id", "rock_seaurchin")
        self.rgb_topic = rospy.get_param(
            root + "rgb_topic", "/%s/rgb/image_color" % self.vehicle_name)
        self.depth_topic = rospy.get_param(
            root + "depth_topic", "/%s/depth/image_depth" % self.vehicle_name)
        self.pose_topic = rospy.get_param(
            root + "pose_topic", "/%s/odometry" % self.vehicle_name)
        self.action_request_topic = rospy.get_param(
            root + "action_request_topic",
            "/underwater_objectnav/action_request",
        )
        self.action_status_topic = rospy.get_param(
            root + "action_status_topic",
            "/underwater_objectnav/action_status",
        )
        self.navigation_status_topic = rospy.get_param(
            root + "navigation_status_topic",
            "/underwater_objectnav/navigation_status",
        )
        self.sync_slop = float(rospy.get_param(root + "sync_slop_s", 0.08))
        self.max_observation_age = float(rospy.get_param(
            root + "max_observation_age_s", 0.25))
        self.inference_interval = float(rospy.get_param(
            root + "inference_interval_s", 2.0))
        self.inference_timeout = float(rospy.get_param(
            root + "inference_timeout_s", 1.8))
        self.max_steps = int(rospy.get_param(root + "max_steps", 100))
        self.target_sensor_z = float(rospy.get_param(
            root + "target_sensor_world_z_m", 3.3))
        self.depth_tolerance = float(rospy.get_param(
            root + "depth_tolerance_m", 0.05))
        self.depth_settle_time = float(rospy.get_param(
            root + "depth_settle_time_s", 0.5))
        self.worker_host = rospy.get_param(root + "worker_host", "127.0.0.1")
        self.worker_port = int(rospy.get_param(root + "worker_port", 29500))
        worker_connect_timeout = float(rospy.get_param(
            root + "worker_connect_timeout_s", 1.0))
        self.record_root = rospy.get_param(root + "record_root")
        self.visualize = bool(rospy.get_param(root + "visualize", True))
        visualization_scale = int(rospy.get_param(
            root + "visualization_scale", 5))
        self.checkpoint = rospy.get_param("~checkpoint")
        self.bc_config = rospy.get_param("~bc_config")
        self.online_config = rospy.get_param("~online_config")
        with open(self.bc_config, encoding="utf-8") as stream:
            bc_document = yaml.safe_load(stream) or {}
        translation = bc_document.get("camera", {}).get(
            "T_base_camera", {}).get("translation_m")
        if not isinstance(translation, list) or len(translation) != 3:
            raise ValueError(
                "bc.yaml camera.T_base_camera.translation_m must contain "
                "three values")
        self.base_to_camera_translation = np.asarray(
            translation, dtype=np.float64)

        if self.inference_interval <= 0.0:
            raise ValueError("inference_interval_s must be positive")
        if self.inference_timeout <= 0.0:
            raise ValueError("inference_timeout_s must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.client = InferenceClient(
            self.worker_host, self.worker_port, worker_connect_timeout)
        self.recorder = OnlineEpisodeRecorder(
            self.record_root,
            self.scene_id,
            self.checkpoint,
            self.bc_config,
            self.online_config,
        )
        self.visualizer = (
            DebugVisualizer(
                "/underwater_objectnav/debug", visualization_scale)
            if self.visualize
            else None
        )

        self.latest_observation = None
        self.worker_ready = False
        self.active = False
        self.recording_active = False
        self.next_unrecorded_episode_id = 1000000000
        self.state = "NOT_READY"
        self.episode_id = 0
        self.goal_category = ""
        self.step_id = 0
        self.next_action_id = 1
        self.next_request_id = 1
        self.navigation_start_time = rospy.Time()
        self.last_action = ""
        self.last_action_start = None
        self.current_action_id = None
        self.current_action = None
        self.inference_busy = False
        self.depth_ready_since = None
        self.abort_reason = None
        self.abort_cancelled = False
        self.policy_stop_pending = False

        self.action_request_pub = rospy.Publisher(
            self.action_request_topic, ActionRequest, queue_size=10)
        self.status_pub = rospy.Publisher(
            self.navigation_status_topic,
            NavigationStatus,
            queue_size=10,
            latch=True,
        )
        rospy.Subscriber(
            self.action_status_topic,
            ActionStatus,
            self.action_status_cb,
            queue_size=20,
        )
        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        pose_sub = message_filters.Subscriber(self.pose_topic, Odometry)
        queue_size = int(rospy.get_param(root + "sync_queue_size", 20))
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, pose_sub], queue_size, self.sync_slop)
        self.sync.registerCallback(self.observation_cb)

        rospy.Service(
            "/underwater_objectnav/start_navigation",
            StartNavigation,
            self.start_navigation,
        )
        rospy.Service(
            "/underwater_objectnav/cancel_navigation",
            CancelNavigation,
            self.cancel_navigation,
        )
        self.timer = rospy.Timer(rospy.Duration(0.05), self.update)
        rospy.on_shutdown(self.shutdown)
        self.publish_status("waiting for Conda inference worker")
        threading.Thread(target=self.connect_worker, daemon=True).start()
        rospy.loginfo(
            "ObjectNav online node: RGB=%s Depth=%s Pose=%s",
            self.rgb_topic,
            self.depth_topic,
            self.pose_topic,
        )

    def connect_worker(self):
        while not rospy.is_shutdown() and not self.client.connected:
            try:
                self.client.connect()
            except Exception as exc:
                rospy.loginfo_throttle(
                    2.0, "waiting for inference worker: %s", exc)
                time.sleep(0.5)
                continue
            with self.lock:
                self.worker_ready = True
                if not self.active:
                    self.state = "IDLE"
                self.publish_status("inference worker ready")
            rospy.loginfo("ObjectNav inference worker connected and READY")

    def observation_cb(self, rgb, depth, odom):
        stamp = max(rgb.header.stamp, depth.header.stamp, odom.header.stamp)
        with self.lock:
            self.latest_observation = (rgb, depth, odom, stamp)

    def observation_is_fresh(self, observation=None):
        observation = observation or self.latest_observation
        if observation is None:
            return False
        age = (rospy.Time.now() - observation[3]).to_sec()
        return -0.05 <= age <= self.max_observation_age

    @staticmethod
    def pose_dict(odom):
        pose = odom.pose.pose
        roll, pitch, yaw = rpy_from_quaternion(pose.orientation)
        return {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        }

    def publish_status(self, message="", failure_reason="", finish_time=None):
        status = NavigationStatus()
        status.header.stamp = rospy.Time.now()
        status.episode_id = int(self.episode_id)
        status.goal_category = self.goal_category
        status.state = self.state
        status.step_id = int(self.step_id)
        status.last_action = self.last_action
        status.failure_reason = failure_reason
        status.message = message
        status.start_time = self.navigation_start_time
        status.finish_time = finish_time or rospy.Time()
        self.status_pub.publish(status)

    def start_navigation(self, request):
        goal = request.goal_category.strip().lower()
        with self.lock:
            if self.active:
                return StartNavigationResponse(
                    False, int(self.episode_id), "navigation already active")
            if goal not in SUPPORTED_GOALS:
                return StartNavigationResponse(
                    False, 0, "unsupported goal_category %r" % goal)
            if not self.worker_ready or not self.client.connected:
                return StartNavigationResponse(
                    False, 0, "inference_worker_not_ready")
            if not self.observation_is_fresh():
                return StartNavigationResponse(
                    False, 0, "fresh synchronized RGB-D-Pose is unavailable")
            try:
                self.client.request(
                    {"type": "START_EPISODE", "goal_category": goal},
                    timeout_s=1.0,
                )
            except Exception as exc:
                self.worker_ready = False
                return StartNavigationResponse(
                    False, 0, "could not start inference episode: %s" % exc)

            now = rospy.Time.now()
            start_pose = self.pose_dict(self.latest_observation[2])
            try:
                episode_id, episode_dir = self.recorder.start(
                    goal, now.to_sec(), start_pose)
            except Exception as exc:
                episode_id = self.next_unrecorded_episode_id
                self.next_unrecorded_episode_id += 1
                episode_dir = "recording unavailable"
                self.recording_active = False
                rospy.logwarn(
                    "ObjectNav recording could not start; navigation "
                    "continues without files: %s", exc)
            else:
                self.recording_active = True

            self.active = True
            self.state = "WAITING_FOR_DEPTH"
            self.episode_id = int(episode_id)
            self.goal_category = goal
            self.step_id = 0
            self.navigation_start_time = now
            self.last_action = ""
            self.last_action_start = None
            self.current_action_id = None
            self.current_action = None
            self.inference_busy = False
            self.depth_ready_since = None
            self.abort_reason = None
            self.abort_cancelled = False
            self.policy_stop_pending = False
            self.publish_status("waiting for target depth")
            rospy.loginfo(
                "ObjectNav episode %d started: goal=%s record=%s",
                self.episode_id,
                goal,
                episode_dir,
            )
            return StartNavigationResponse(
                True, self.episode_id, "navigation accepted")

    def cancel_navigation(self, _request):
        with self.lock:
            if not self.active:
                return CancelNavigationResponse(False, "no active navigation")
            self.begin_abort("cancelled", cancelled=True)
            return CancelNavigationResponse(True, "cancellation accepted")

    def update_depth_readiness(self, now, pose):
        tx, ty, tz = self.base_to_camera_translation
        # Third row of R_world_base (ZYX convention). Yaw does not affect z.
        sensor_z = float(
            pose["z"]
            - math.sin(pose["pitch"]) * tx
            + math.cos(pose["pitch"]) * math.sin(pose["roll"]) * ty
            + math.cos(pose["pitch"]) * math.cos(pose["roll"]) * tz
        )
        error = abs(sensor_z - self.target_sensor_z)
        if error > self.depth_tolerance:
            self.depth_ready_since = None
            rospy.loginfo_throttle(
                2.0,
                "ObjectNav waiting for sensor z: current=%.3f target=%.3f",
                sensor_z,
                self.target_sensor_z,
            )
            return False
        if self.depth_ready_since is None:
            self.depth_ready_since = now
            return self.depth_settle_time <= 0.0
        return (now - self.depth_ready_since).to_sec() >= self.depth_settle_time

    def update(self, _event):
        with self.lock:
            if not self.active:
                return
            now = rospy.Time.now()
            if self.state != "ABORTING" and not self.observation_is_fresh():
                self.begin_abort("sensor_timeout")
                return
            if self.state == "WAITING_FOR_DEPTH":
                pose = self.pose_dict(self.latest_observation[2])
                if self.update_depth_readiness(now, pose):
                    self.state = "RUNNING"
                    self.publish_status("target depth reached")
                else:
                    return
            if self.state != "RUNNING":
                return
            if self.current_action_id is not None or self.inference_busy:
                return
            if self.last_action_start is not None:
                due = self.last_action_start + rospy.Duration(
                    self.inference_interval)
                if now < due:
                    return
            self.start_inference()

    def convert_observation(self, observation):
        rgb_msg, depth_msg, odom, stamp = observation
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = np.asarray(
            self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"),
            dtype=np.float32,
        )
        return {
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "depth": depth,
            "pose": self.pose_dict(odom),
            "stamp": stamp,
        }

    def start_inference(self):
        if not self.observation_is_fresh():
            self.begin_abort("sensor_timeout")
            return
        try:
            sample = self.convert_observation(self.latest_observation)
        except (CvBridgeError, ValueError, TypeError) as exc:
            rospy.logerr("ObjectNav sensor conversion failed: %s", exc)
            self.begin_abort("sensor_conversion_error")
            return
        request_id = self.next_request_id
        self.next_request_id += 1
        self.inference_busy = True
        self.state = "INFERENCING"
        inference_start = rospy.Time.now()
        self.publish_status("running GFL, mapping and BC")
        thread = threading.Thread(
            target=self.inference_task,
            args=(request_id, sample, inference_start),
            daemon=True,
        )
        thread.start()

    def inference_task(self, request_id, sample, inference_start):
        payload = {
            "type": "INFER",
            "request_id": int(request_id),
            "observation_time": sample["stamp"].to_sec(),
            "pose": sample["pose"],
        }
        try:
            response, arrays = self.client.request(
                payload,
                arrays={"rgb": sample["rgb"], "depth": sample["depth"]},
                timeout_s=self.inference_timeout,
            )
            if int(response.get("request_id", -1)) != int(request_id):
                raise RuntimeError("inference request_id mismatch")
        except Exception as exc:
            rospy.logerr("ObjectNav inference failed: %s", exc)
            with self.lock:
                self.inference_busy = False
                self.worker_ready = False
                self.begin_abort("inference_error")
            return
        inference_finish = rospy.Time.now()
        with self.lock:
            self.inference_busy = False
            if not self.active:
                return
            if self.abort_reason is not None:
                self.publish_abort_stop()
                return
            try:
                self.handle_inference_result(
                    sample, response, arrays,
                    inference_start, inference_finish)
            except Exception as exc:
                rospy.logerr(
                    "ObjectNav received an invalid inference result: %s", exc)
                self.begin_abort("invalid_inference_result")

    def handle_inference_result(
        self, sample, result, arrays, inference_start, inference_finish
    ):
        action = str(result["action"]).strip().upper()
        if action not in ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"):
            rospy.logerr("BC returned unsupported action %r", action)
            self.begin_abort("invalid_policy_action")
            return
        global_map = arrays["global_map"]
        local_map = arrays["local_map"]
        action_id = self.next_action_id
        self.next_action_id += 1
        action_request_time = rospy.Time.now()
        if self.recording_active:
            try:
                self.recorder.record_decision(
                    self.step_id,
                    action_id,
                    sample["stamp"].to_sec(),
                    inference_start.to_sec(),
                    inference_finish.to_sec(),
                    action_request_time.to_sec(),
                    sample["pose"],
                    action,
                    result,
                    sample["rgb"],
                    sample["depth"],
                    global_map,
                    local_map,
                )
            except Exception as exc:
                rospy.logwarn(
                    "ObjectNav recording failed but navigation continues: %s",
                    exc)
        if self.visualizer is not None:
            try:
                self.visualizer.submit(
                    sample["stamp"], self.step_id, sample["rgb"], result,
                    global_map, local_map)
            except Exception as exc:
                rospy.logwarn(
                    "ObjectNav visualization skipped one frame: %s", exc)

        message = ActionRequest()
        message.action_id = action_id
        message.action = action
        message.request_time = action_request_time
        self.current_action_id = action_id
        self.current_action = action
        self.last_action = action
        self.policy_stop_pending = action == "STOP"
        self.state = "WAITING_FOR_ACTION"
        self.step_id += 1
        self.action_request_pub.publish(message)
        self.publish_status(
            "published action %s probabilities=%s" %
            (action, [round(value, 4) for value in result["probabilities"]]))
        rospy.loginfo(
            "ObjectNav episode=%d step=%d action=%s probabilities=%s infer=%.3fs",
            self.episode_id,
            self.step_id - 1,
            action,
            [round(value, 4) for value in result["probabilities"]],
            result["inference_time_s"],
        )

    def action_status_cb(self, message):
        with self.lock:
            if not self.active or message.action_id != self.current_action_id:
                return
            start_time = ros_time_seconds(message.start_time)
            finish_time = ros_time_seconds(message.finish_time)
            if self.recording_active:
                self.recorder.update_action_status(
                    message.action_id, message.state, start_time, finish_time)
            if message.state == "STARTED":
                self.last_action_start = (
                    message.start_time
                    if message.start_time != rospy.Time()
                    else rospy.Time.now()
                )
                self.publish_status("action %s started" % message.action)
                return
            if message.state == "SUCCEEDED":
                self.current_action_id = None
                self.current_action = None
                if self.abort_reason is not None:
                    self.publish_abort_stop()
                elif self.step_id >= self.max_steps:
                    self.begin_abort("max_steps_exceeded")
                else:
                    self.state = "RUNNING"
                    self.publish_status("action succeeded")
                return
            if message.state == "TIMEOUT":
                self.current_action_id = None
                self.current_action = None
                if self.abort_reason is None:
                    self.abort_reason = "action_timeout"
                self.publish_abort_stop()
                return
            if message.state == "STOPPED" and self.current_action == "STOP":
                self.current_action_id = None
                self.current_action = None
                if self.policy_stop_pending and self.abort_reason is None:
                    self.finish_navigation(True, None, "SUCCEEDED")
                else:
                    terminal = "CANCELLED" if self.abort_cancelled else "FAILED"
                    self.finish_navigation(False, self.abort_reason, terminal)

    def begin_abort(self, reason, cancelled=False):
        if not self.active:
            return
        if self.abort_reason is None:
            self.abort_reason = str(reason)
        self.abort_cancelled = self.abort_cancelled or bool(cancelled)
        self.state = "ABORTING"
        self.publish_status("stopping navigation", self.abort_reason)
        rospy.logerr("ObjectNav aborting episode %d: %s",
                     self.episode_id, self.abort_reason)
        if not self.inference_busy and self.current_action_id is None:
            self.publish_abort_stop()

    def publish_abort_stop(self):
        if not self.active or self.current_action_id is not None:
            return
        action_id = self.next_action_id
        self.next_action_id += 1
        request = ActionRequest()
        request.action_id = action_id
        request.action = "STOP"
        request.request_time = rospy.Time.now()
        self.current_action_id = action_id
        self.current_action = "STOP"
        self.last_action = "STOP"
        self.policy_stop_pending = False
        self.action_request_pub.publish(request)
        self.publish_status("abort STOP published", self.abort_reason)

    def finish_navigation(self, success, failure_reason, terminal_state):
        finish_time = rospy.Time.now()
        try:
            self.client.request({"type": "RESET"}, timeout_s=1.0)
        except Exception as exc:
            self.worker_ready = False
            rospy.logwarn("ObjectNav worker reset failed: %s", exc)
        errors = []
        if self.recording_active:
            try:
                errors = self.recorder.finish(
                    success, failure_reason, finish_time.to_sec())
            except Exception as exc:
                errors = ["finalize recording: %s: %s" %
                          (type(exc).__name__, exc)]
        self.recording_active = False
        self.active = False
        self.state = terminal_state
        message = "navigation finished"
        if errors:
            message += "; recording errors: " + "; ".join(errors)
            rospy.logwarn("ObjectNav recording was incomplete: %s", errors)
        self.publish_status(message, failure_reason or "", finish_time)
        if success:
            rospy.loginfo("ObjectNav episode %d SUCCEEDED", self.episode_id)
        else:
            rospy.logerr(
                "ObjectNav episode %d %s: %s",
                self.episode_id,
                terminal_state,
                failure_reason,
            )

    def shutdown(self):
        with self.lock:
            if self.active:
                request = ActionRequest()
                request.action_id = self.next_action_id
                request.action = "STOP"
                request.request_time = rospy.Time.now()
                self.action_request_pub.publish(request)
                if self.recording_active:
                    try:
                        self.recorder.finish(
                            False, "node_shutdown", rospy.Time.now().to_sec())
                    except Exception as exc:
                        rospy.logerr(
                            "could not finalize online recording: %s", exc)
                    self.recording_active = False
                self.active = False
        self.client.close()
        self.recorder.shutdown()


if __name__ == "__main__":
    rospy.init_node("objectnav_online")
    OnlineNavigationNode()
    rospy.spin()
