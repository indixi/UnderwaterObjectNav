#!/usr/bin/env python3
"""水下 ObjectNav 专家示范数据采集节点。

采集时序固定为：同步观测 -> 接收 robot_control 键盘动作 -> 保存观测和动作。
节点只把 RGB、Depth、Pose 和专家动作写入 trajectory.csv；仿真真值位置
仅写入 episode.yaml，因而不会意外成为策略网络的输入。
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
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String

from data_generate.srv import (CancelEpisode, CancelEpisodeResponse,
                               EndEpisode, EndEpisodeResponse,
                               StartEpisode, StartEpisodeResponse)


ACTIONS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def wrap(angle):
    """将角度归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    """从 ROS 四元数提取平面 yaw。"""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def rpy_from_quaternion(q):
    """从 ROS 四元数提取 roll、pitch、yaw，便于完整记录机器人姿态。"""
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch, yaw_from_quaternion(q)


def quaternion_from_yaw(yaw):
    """生成只包含 yaw 的 ROS 四元数。"""
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def yaml_scalar(value):
    """把简单字符串写成安全的 YAML 标量。"""
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
        self.active = False
        self.executing_action = None
        self.action_started = None
        self.action_target = None
        # robot_control 延迟发布期间可能产生多个动作，这里按到达顺序排队。
        self.pending_actions = deque()
        self.episode_dir = None
        self.episode_id = None
        self.step_id = 0
        self.csv_file = None
        self.csv_writer = None

        vehicle = rospy.get_param("~vehicle_name", "bricsbot")
        self.rgb_topic = rospy.get_param("~rgb_topic", "/%s/rgb/image_color" % vehicle)
        self.depth_topic = rospy.get_param("~depth_topic", "/%s/depth/image_depth" % vehicle)
        self.pose_topic = rospy.get_param("~pose_topic", "/%s/odometry" % vehicle)
        self.reference_topic = rospy.get_param("~reference_topic", "/aquaflow/nominal_pose")
        self.action_topic = rospy.get_param("~expert_action_topic",
                                             "/underwater_objectnav/expert_action")
        self.status_topic = rospy.get_param("~status_topic",
                                             "/underwater_objectnav/collector_status")
        default_root = os.path.join(rospy.get_param("~package_path", os.getcwd()),
                                    "underwater_objectnav_dataset")
        self.dataset_root = os.path.abspath(rospy.get_param("~dataset_root", "") or default_root)
        self.slop = float(rospy.get_param("~sync_slop_s", 0.08))
        self.max_age = float(rospy.get_param("~max_observation_age_s", 0.25))
        self.pos_tolerance = float(rospy.get_param("~forward_position_tolerance_m", 0.035))
        self.yaw_tolerance = math.radians(float(rospy.get_param("~turn_angle_tolerance_deg", 3.0)))
        self.action_timeout = float(rospy.get_param("~action_timeout_s", 15.0))
        self.auto_end_on_stop = bool(rospy.get_param("~auto_end_on_stop", False))

        queue_size = int(rospy.get_param("~sync_queue_size", 20))
        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        pose_sub = message_filters.Subscriber(self.pose_topic, Odometry)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, pose_sub], queue_size, self.slop)
        self.sync.registerCallback(self.observation_cb)

        # 目标位姿由 robot_control 的 keyboard_reference_publisher 发布；
        # 本节点只缓存并监视该目标，不再重复发布控制目标。
        self.latest_reference = None
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10, latch=True)
        rospy.Subscriber(self.action_topic, String, self.action_cb, queue_size=10)
        rospy.Subscriber(self.reference_topic, PoseStamped,
                         self.reference_cb, queue_size=10)
        rospy.Service("~start_episode", StartEpisode, self.start_episode)
        rospy.Service("~end_episode", EndEpisode, self.end_episode)
        rospy.Service("~cancel_episode", CancelEpisode, self.cancel_episode)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.check_action)
        rospy.on_shutdown(self.shutdown)

        self.ensure_metadata()
        self.publish_status("IDLE")
        rospy.loginfo("ObjectNav 采集器已启动，RGB=%s，Depth=%s，Pose=%s",
                      self.rgb_topic, self.depth_topic, self.pose_topic)

    def ensure_metadata(self):
        """创建数据集根目录和总 metadata.yaml。"""
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
        """保存最新同步观测；真正写盘延迟到动作回调，保证标签时序正确。"""
        stamp = max(rgb.header.stamp, depth.header.stamp, odom.header.stamp)
        with self.lock:
            self.latest = (rgb, depth, odom, stamp)

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))

    def reference_cb(self, msg):
        """缓存 robot_control 键盘发布器生成的实际目标位姿。"""
        with self.lock:
            self.latest_reference = msg

    def start_episode(self, req):
        with self.lock:
            if self.active:
                return StartEpisodeResponse(False, -1, self.episode_dir or "", "已有 Episode 正在记录")
            if not req.goal_category:
                return StartEpisodeResponse(False, -1, "", "goal_category 不能为空")
            existing = [name for name in os.listdir(self.dataset_root)
                        if name.startswith("episode_") and os.path.isdir(os.path.join(self.dataset_root, name))]
            numbers = [int(name.split("_")[1]) for name in existing
                       if name.split("_")[1].isdigit()]
            self.episode_id = max(numbers, default=0) + 1
            self.episode_dir = os.path.join(self.dataset_root, "episode_%04d" % self.episode_id)
            os.makedirs(os.path.join(self.episode_dir, "rgb"))
            os.makedirs(os.path.join(self.episode_dir, "depth"))
            self.csv_file = open(os.path.join(self.episode_dir, "trajectory.csv"), "w",
                                 newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["step_id", "timestamp", "rgb_path", "depth_path",
                                      "x", "y", "z", "roll", "pitch", "yaw",
                                      "goal_category", "expert_action"])
            self.active = True
            self.executing_action = None
            self.step_id = 0
            self.write_episode_yaml(req, None, None, False, None)
            self.publish_status("RECORDING episode_%04d" % self.episode_id)
            return StartEpisodeResponse(True, self.episode_id, self.episode_dir, "Episode 已开始")

    def action_cb(self, msg):
        action = msg.data.strip().upper()
        if action not in ACTIONS:
            rospy.logwarn("忽略未知专家动作 %r，可选：%s", msg.data, ", ".join(ACTIONS))
            return
        with self.lock:
            if not self.active:
                rospy.logwarn("当前没有活动 Episode，动作 %s 未记录", action)
                return
            if action != "STOP" and self.latest_reference is None:
                rospy.logwarn("尚未收到 robot_control 的目标位姿，动作 %s 未记录", action)
                return
            # 目标和动作先入队；只有前一个动作完成后，动作真正开始时
            # 才保存观测，保证数据与实际执行时刻一致。
            target = self.latest_reference
            self.pending_actions.append((action, target))
            self.start_next_action()

    def start_next_action(self):
        """若当前空闲，则启动队列中的下一个原子动作。"""
        if self.executing_action is not None or not self.pending_actions:
            return
        action, target = self.pending_actions.popleft()
        if action != "STOP" and target is None:
            rospy.logwarn("动作 %s 缺少目标位姿，跳过执行", action)
            self.start_next_action()
            return
        if self.latest is None or \
                (rospy.Time.now() - self.latest[3]).to_sec() > self.max_age:
            rospy.logwarn("动作 %s 开始时没有新鲜观测，跳过该动作", action)
            self.start_next_action()
            return
        if action == "STOP":
            self.publish_status("STOP_RECORDED")
            try:
                self.save_step(action, self.latest)
            except (CvBridgeError, IOError, OSError, ValueError) as exc:
                rospy.logerr("保存 STOP step 失败：%s", exc)
            if self.auto_end_on_stop:
                self.finish_episode(True, None)
            else:
                self.start_next_action()
            return
        try:
            self.save_step(action, self.latest)
        except (CvBridgeError, IOError, OSError, ValueError) as exc:
            rospy.logerr("保存 step 失败，动作不会执行：%s", exc)
            self.start_next_action()
            return
        self.action_target = target
        self.executing_action = action
        self.action_started = rospy.Time.now()
        self.publish_status("EXECUTING %s" % action)

    def save_step(self, action, sample):
        """将动作执行前的同步观测保存为一个连续 step。"""
        rgb_msg, depth_msg, odom_msg, stamp = sample
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth, dtype=np.float32)
        index = "%06d" % self.step_id
        rgb_rel = os.path.join("rgb", index + ".png")
        depth_rel = os.path.join("depth", index + ".npy")
        if not cv2.imwrite(os.path.join(self.episode_dir, rgb_rel), rgb):
            raise IOError("OpenCV 无法写入 RGB 图像")
        np.save(os.path.join(self.episode_dir, depth_rel), depth)
        pose = odom_msg.pose.pose
        roll, pitch, yaw = rpy_from_quaternion(pose.orientation)
        timestamp = stamp.to_sec() if stamp != rospy.Time() else rospy.Time.now().to_sec()
        self.csv_writer.writerow([self.step_id, "%.9f" % timestamp, rgb_rel, depth_rel,
                                  "%.9f" % pose.position.x, "%.9f" % pose.position.y,
                                  "%.9f" % pose.position.z, "%.9f" % roll,
                                  "%.9f" % pitch, "%.9f" % yaw,
                                  self.goal_category, action])
        self.csv_file.flush()
        self.step_id += 1

    def check_action(self, _event):
        with self.lock:
            if not self.active or self.executing_action is None or self.latest is None:
                return
            current = self.latest[2].pose.pose
            target = self.action_target.pose
            dx = target.position.x - current.position.x
            dy = target.position.y - current.position.y
            distance = math.hypot(dx, dy)
            yaw_error = abs(wrap(yaw_from_quaternion(target.orientation) -
                                 yaw_from_quaternion(current.orientation)))
            done = (distance <= self.pos_tolerance if self.executing_action == "FORWARD"
                    else yaw_error <= self.yaw_tolerance)
            elapsed = (rospy.Time.now() - self.action_started).to_sec()
            if done:
                self.publish_status("READY_FOR_NEXT_STEP")
                self.executing_action = None
                self.action_target = None
                self.start_next_action()
            elif elapsed > self.action_timeout:
                self.finish_episode(False, "action_timeout_%s" % self.executing_action.lower())

    def end_episode(self, req):
        with self.lock:
            if not self.active:
                return EndEpisodeResponse(False, "当前没有活动 Episode")
            self.finish_episode(req.success, req.failure_reason or (None if req.success else "expert_failure"))
            return EndEpisodeResponse(True, "Episode 已结束")

    def cancel_episode(self, _req):
        with self.lock:
            if not self.active:
                return CancelEpisodeResponse(False, "当前没有活动 Episode")
            directory = self.episode_dir
            self.close_files()
            shutil.rmtree(directory, ignore_errors=False)
            self.reset_state()
            self.publish_status("CANCELLED")
            return CancelEpisodeResponse(True, "已取消并删除 %s" % directory)

    def finish_episode(self, success, failure_reason):
        self.write_episode_yaml(None, success, failure_reason, True, self.start_request)
        self.close_files()
        self.reset_state()
        self.publish_status("FINISHED")

    def write_episode_yaml(self, request, success, failure_reason, final, saved_request):
        """写入 episode.yaml；初始写入和结束写入均使用同一格式。"""
        if request is not None:
            self.start_request = request
            goal = request.goal_category
            scene = request.scene_id or "unknown"
            start_pose = self.latest[2].pose.pose if self.latest else None
        else:
            request = saved_request
            goal = self.goal_category
            scene = self.scene_id
            start_pose = self.start_pose
        if request is not None and not hasattr(self, "goal_category"):
            self.goal_category, self.scene_id = goal, scene
            self.start_pose = start_pose
        path = os.path.join(self.episode_dir, "episode.yaml")
        pose = start_pose or self.start_pose
        with open(path, "w", encoding="utf-8") as out:
            out.write("episode_id: %d\nscene_id: %s\ngoal_category: %s\n" %
                      (self.episode_id, yaml_scalar(scene), yaml_scalar(goal)))
            out.write("success: %s\nfailure_reason: %s\n" %
                      (yaml_scalar(success), yaml_scalar(failure_reason)))
            out.write("start_pose:\n")
            for name, value in (("x", pose.position.x if pose else 0.0),
                                ("y", pose.position.y if pose else 0.0),
                                ("z", pose.position.z if pose else 0.0),
                                ("yaw", yaw_from_quaternion(pose.orientation) if pose else 0.0)):
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
        self.executing_action = None
        self.action_target = None
        self.pending_actions.clear()
        self.action_started = None
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
