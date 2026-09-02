#!/usr/bin/env python3
"""Event-driven RGB/depth/pose collector for ObjectNav.

The collector never schedules or completes actions.  It listens for an
ActionStatus STARTED event and selects the synchronized sensor sample closest
to the action start timestamp.
"""
import csv
import math
import os
import shutil
import threading
from collections import deque

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from data_generate.srv import (CancelEpisode, CancelEpisodeResponse,
                               EndEpisode, EndEpisodeResponse,
                               StartEpisode, StartEpisodeResponse)
from robot_control.msg import ActionStatus


ACTIONS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def rpy_from_quaternion(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch, yaw_from_quaternion(q)


def yaml_scalar(value):
    if value is None:
        return "null"
    text = str(value)
    if text.lower() in ("true", "false") or text == "null":
        return text.lower()
    try:
        float(text)
        return text
    except ValueError:
        return "'" + text.replace("'", "''") + "'"


class ObjectNavDataCollector:
    def __init__(self):
        self.lock = threading.RLock()
        self.bridge = CvBridge()
        self.latest = None
        self.observation_buffer = deque(maxlen=300)
        self.active = False
        self.episode_start_time = None
        self.recorded_action_ids = set()
        self.episode_dir = None
        self.episode_id = None
        self.step_id = 0
        self.csv_file = None
        self.csv_writer = None
        self.start_request = None
        self.start_pose = None
        self.goal_category = None
        self.scene_id = None

        vehicle = rospy.get_param("~vehicle_name", "bricsbot")
        self.rgb_topic = rospy.get_param("~rgb_topic", "/%s/rgb/image_color" % vehicle)
        self.depth_topic = rospy.get_param("~depth_topic", "/%s/depth/image_depth" % vehicle)
        self.pose_topic = rospy.get_param("~pose_topic", "/%s/odometry" % vehicle)
        self.action_status_topic = rospy.get_param(
            "~action_status_topic", "/underwater_objectnav/action_status")
        default_root = os.path.join(rospy.get_param("~package_path", os.getcwd()),
                                    "underwater_objectnav_dataset")
        self.dataset_root = os.path.abspath(
            rospy.get_param("~dataset_root", "") or default_root)
        self.slop = float(rospy.get_param("~sync_slop_s", 0.08))
        self.max_age = float(rospy.get_param("~max_observation_age_s", 0.25))
        self.auto_end_on_stop = bool(
            rospy.get_param("~auto_end_on_stop", True))

        queue_size = int(rospy.get_param("~sync_queue_size", 20))
        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        pose_sub = message_filters.Subscriber(self.pose_topic, Odometry)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, pose_sub], queue_size, self.slop)
        self.sync.registerCallback(self.observation_cb)
        rospy.Subscriber(self.action_status_topic, ActionStatus,
                         self.action_status_cb, queue_size=20)
        rospy.Service("~start_episode", StartEpisode, self.start_episode)
        rospy.Service("~end_episode", EndEpisode, self.end_episode)
        rospy.Service("~cancel_episode", CancelEpisode, self.cancel_episode)
        rospy.on_shutdown(self.shutdown)

        self.ensure_metadata()
        rospy.loginfo("ObjectNav collector started: RGB=%s Depth=%s Pose=%s",
                      self.rgb_topic, self.depth_topic, self.pose_topic)

    def ensure_metadata(self):
        os.makedirs(self.dataset_root, exist_ok=True)
        path = os.path.join(self.dataset_root, "metadata.yaml")
        if os.path.exists(path):
            return
        with open(path, "w", encoding="utf-8") as out:
            out.write("dataset_name: underwater_objectnav_bc\n")
            out.write("goal_categories:\n  - sea_urchin\n")
            out.write("actions:\n  0: FORWARD\n  1: TURN_LEFT\n  2: TURN_RIGHT\n  3: STOP\n")
            out.write("camera:\n  rgb_topic: %s\n  depth_topic: %s\n" %
                      (yaml_scalar(self.rgb_topic), yaml_scalar(self.depth_topic)))
            out.write("  intrinsic_file: camera_intrinsic.yaml\n")
            out.write("  extrinsic_file: camera_extrinsic.yaml\n")

    def observation_cb(self, rgb, depth, odom):
        stamp = max(rgb.header.stamp, depth.header.stamp, odom.header.stamp)
        with self.lock:
            sample = (rgb, depth, odom, stamp)
            self.latest = sample
            self.observation_buffer.append(sample)

    def action_status_cb(self, msg):
        if msg.state != "STARTED":
            return
        with self.lock:
            if not self.active:
                # Idle keyboard operation is normal. The collector is only an
                # optional observer and must never interfere with execution.
                return
            if self.episode_start_time is not None and \
                    msg.start_time != rospy.Time() and \
                    msg.start_time < self.episode_start_time:
                return
            if msg.action_id in self.recorded_action_ids:
                return
            action = msg.action.strip().upper()
            if action not in ACTIONS:
                rospy.logwarn("ignore unknown action status %r", msg.action)
                return
            sample = self.sample_near(msg.start_time)
            if sample is None:
                rospy.logwarn("no synchronized observation near action id=%d start time",
                              msg.action_id)
                if action == "STOP" and self.auto_end_on_stop:
                    self.finish_episode(False, "stop_observation_missing")
                return
            try:
                self.save_step(action, sample, msg.action_id, msg.start_time)
                self.recorded_action_ids.add(msg.action_id)
            except (CvBridgeError, IOError, OSError, ValueError) as exc:
                rospy.logerr("save step failed for action id=%d: %s",
                             msg.action_id, exc)
                if action == "STOP" and self.auto_end_on_stop:
                    self.finish_episode(False, "stop_observation_save_failed")
                return
            if action == "STOP" and self.auto_end_on_stop:
                rospy.loginfo(
                    "STOP action id=%d saved as final step; ending Episode successfully",
                    msg.action_id)
                self.finish_episode(True, None)

    def sample_near(self, stamp):
        if not self.observation_buffer:
            return None
        if stamp == rospy.Time():
            return self.latest
        sample = min(self.observation_buffer,
                     key=lambda item: abs((item[3] - stamp).to_sec()))
        error = abs((sample[3] - stamp).to_sec())
        if self.max_age > 0.0 and error > self.max_age:
            return None
        return sample

    def start_episode(self, req):
        with self.lock:
            if self.active:
                return StartEpisodeResponse(False, -1, self.episode_dir or "",
                                            "已有 Episode 正在记录")
            if not req.goal_category:
                return StartEpisodeResponse(False, -1, "", "goal_category 不能为空")
            existing = [name for name in os.listdir(self.dataset_root)
                        if name.startswith("episode_") and
                        os.path.isdir(os.path.join(self.dataset_root, name))]
            numbers = [int(name.split("_")[1]) for name in existing
                       if name.split("_")[1].isdigit()]
            self.episode_id = max(numbers, default=0) + 1
            self.episode_dir = os.path.join(
                self.dataset_root, "episode_%04d" % self.episode_id)
            os.makedirs(os.path.join(self.episode_dir, "rgb"))
            os.makedirs(os.path.join(self.episode_dir, "depth"))
            self.csv_file = open(os.path.join(self.episode_dir, "trajectory.csv"),
                                 "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "step_id", "action_id", "action_start_time", "timestamp",
                "rgb_path", "depth_path", "x", "y", "z", "roll", "pitch",
                "yaw", "goal_category", "expert_action"])
            self.episode_start_time = rospy.Time.now()
            self.active = True
            self.step_id = 0
            self.recorded_action_ids.clear()
            self.write_episode_yaml(req, None, None, False, None)
            return StartEpisodeResponse(True, self.episode_id,
                                        self.episode_dir, "Episode 已开始")

    def save_step(self, action, sample, action_id, action_start_time):
        rgb_msg, depth_msg, odom_msg, stamp = sample
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = np.asarray(
            self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"),
            dtype=np.float32)
        index = "%06d" % self.step_id
        rgb_rel = os.path.join("rgb", index + ".png")
        depth_rel = os.path.join("depth", index + ".npy")
        if not cv2.imwrite(os.path.join(self.episode_dir, rgb_rel), rgb):
            raise IOError("OpenCV 无法写入 RGB 图像")
        np.save(os.path.join(self.episode_dir, depth_rel), depth)
        pose = odom_msg.pose.pose
        roll, pitch, yaw = rpy_from_quaternion(pose.orientation)
        self.csv_writer.writerow([
            self.step_id, action_id, "%.9f" % action_start_time.to_sec(),
            "%.9f" % stamp.to_sec(), rgb_rel, depth_rel,
            "%.9f" % pose.position.x, "%.9f" % pose.position.y,
            "%.9f" % pose.position.z, "%.9f" % roll, "%.9f" % pitch,
            "%.9f" % yaw, self.goal_category, action])
        self.csv_file.flush()
        self.step_id += 1

    def end_episode(self, req):
        with self.lock:
            if not self.active:
                return EndEpisodeResponse(False, "当前没有活动 Episode")
            self.finish_episode(req.success,
                                req.failure_reason or
                                (None if req.success else "expert_failure"))
            return EndEpisodeResponse(True, "Episode 已结束")

    def cancel_episode(self, _req):
        with self.lock:
            if not self.active:
                return CancelEpisodeResponse(False, "当前没有活动 Episode")
            directory = self.episode_dir
            self.close_files()
            shutil.rmtree(directory, ignore_errors=False)
            self.reset_state()
            return CancelEpisodeResponse(True, "已取消并删除 %s" % directory)

    def finish_episode(self, success, failure_reason):
        self.write_episode_yaml(None, success, failure_reason, True,
                                self.start_request)
        self.close_files()
        self.reset_state()

    def write_episode_yaml(self, request, success, failure_reason, final,
                           saved_request):
        if request is not None:
            self.start_request = request
            self.goal_category = request.goal_category
            self.scene_id = request.scene_id or "unknown"
            self.start_pose = self.latest[2].pose.pose if self.latest else None
        else:
            request = saved_request
        path = os.path.join(self.episode_dir, "episode.yaml")
        pose = self.start_pose
        with open(path, "w", encoding="utf-8") as out:
            out.write("episode_id: %d\nscene_id: %s\ngoal_category: %s\n" %
                      (self.episode_id, yaml_scalar(self.scene_id),
                       yaml_scalar(self.goal_category)))
            out.write("success: %s\nfailure_reason: %s\n" %
                      (yaml_scalar(success), yaml_scalar(failure_reason)))
            out.write("start_pose:\n")
            for name, value in (("x", pose.position.x if pose else 0.0),
                                ("y", pose.position.y if pose else 0.0),
                                ("z", pose.position.z if pose else 0.0),
                                ("yaw", yaw_from_quaternion(pose.orientation)
                                 if pose else 0.0)):
                out.write("  %s: %.9f\n" % (name, value))
            if final:
                out.write("step_count: %d\n" % self.step_id)

    def close_files(self):
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None

    def reset_state(self):
        self.active = False
        self.episode_start_time = None
        self.recorded_action_ids.clear()
        self.episode_dir = None
        self.episode_id = None
        self.step_id = 0
        self.start_request = None
        self.start_pose = None
        self.goal_category = None
        self.scene_id = None

    def shutdown(self):
        with self.lock:
            if self.active:
                self.finish_episode(False, "node_shutdown")


if __name__ == "__main__":
    rospy.init_node("objectnav_data_collector")
    ObjectNavDataCollector()
    rospy.spin()
