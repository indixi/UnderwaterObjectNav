#!/usr/bin/env python3
"""Keyboard point-reference publisher for position-mode PID control.

W: create a target 0.5 m ahead of the current vehicle pose.
A/D: create a target with +/-30 deg yaw at the current position.
Space: hold the current pose and publish the ObjectNav STOP label.

The node publishes PoseStamped references and a discrete expert-action label;
it does not publish PWM or force commands directly. Each key action is computed
from the latest odometry.
"""
import math
import select
import sys
import termios
import tty
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


class KeyboardReferencePublisher:
    def __init__(self):
        root = "~keyboard_reference_publisher/"
        self.vehicle_name = rospy.get_param(root + "vehicle_name", "bricsbot")
        self.odom_topic = rospy.get_param(
            root + "odom_topic", "/%s/odometry" % self.vehicle_name)
        self.reference_topic = rospy.get_param(
            root + "reference_topic", "/aquaflow/nominal_pose")
        self.action_topic = rospy.get_param(
            root + "action_topic", "/underwater_objectnav/expert_action")
        self.frame_id = rospy.get_param(root + "frame_id", "world_ned")
        self.publish_rate_hz = max(
            1.0, float(rospy.get_param(root + "publish_rate_hz", 20.0)))
        self.forward_distance = float(
            rospy.get_param(root + "forward_distance_m", 0.5))
        self.turn_angle = math.radians(float(
            rospy.get_param(root + "turn_angle_deg", 30.0)))
        self.odom_timeout = max(
            0.0, float(rospy.get_param(root + "odom_timeout_s", 0.25)))
        self.key_repeat_guard = max(
            0.0, float(rospy.get_param(root + "key_repeat_guard_s", 0.10)))
        # 两次专家动作之间的延迟，供数据采集节点使用同一个动作事件。
        self.command_delay = max(
            0.0, float(rospy.get_param(root + "command_delay_s", 5.0)))
        self.initial_command_delay = max(
            0.0, float(rospy.get_param(root + "initial_command_delay_s", 0.0)))

        self.odom = None
        self.target = None
        self.last_key_time = rospy.Time(0)
        self.pending_commands = deque()
        self.has_scheduled_command = False
        self.last_scheduled_time = None
        self.reference_pub = rospy.Publisher(
            self.reference_topic, PoseStamped, queue_size=1)
        self.action_pub = rospy.Publisher(
            self.action_topic, String, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry,
                         self.odom_cb, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate_hz), self.publish)
        rospy.on_shutdown(self.shutdown)

    def odom_cb(self, msg):
        self.odom = msg

    def current_pose(self):
        if self.odom is None:
            return None
        stamp = self.odom.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()
        if self.odom_timeout > 0.0 and \
                (rospy.Time.now() - stamp).to_sec() > self.odom_timeout:
            rospy.logwarn_throttle(2.0, "odometry is too old; key ignored")
            return None
        return self.odom.pose.pose

    def make_target(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        q = quat_from_yaw(yaw)
        msg.pose.orientation.x, msg.pose.orientation.y = q[0], q[1]
        msg.pose.orientation.z, msg.pose.orientation.w = q[2], q[3]
        return msg

    def target_for_action(self, action, base_pose):
        """在动作真正执行时，基于最新实际位姿生成目标。"""
        x = base_pose.position.x
        y = base_pose.position.y
        z = base_pose.position.z
        yaw = yaw_from_quat(base_pose.orientation)
        target_yaw = yaw
        target_x, target_y = x, y

        if action == "FORWARD":
            target_x += self.forward_distance * math.cos(yaw)
            target_y += self.forward_distance * math.sin(yaw)
        elif action == "TURN_LEFT":
            target_yaw += self.turn_angle
        elif action == "TURN_RIGHT":
            target_yaw -= self.turn_angle

        return self.make_target(target_x, target_y, z, target_yaw)

    def process_key(self, key):
        if key not in ("w", "a", "d", " "):
            return
        now = rospy.Time.now()
        if (now - self.last_key_time).to_sec() < self.key_repeat_guard:
            return
        current = self.current_pose()
        if current is None:
            return

        self.last_key_time = now
        # Space 在 ObjectNav 数据集中对应 STOP；控制目标仍保持当前位置。
        action = {"w": "FORWARD", "a": "TURN_LEFT", "d": "TURN_RIGHT",
                  " ": "STOP"}[key]
        if not self.has_scheduled_command:
            delay = self.initial_command_delay
            due_time = now + rospy.Duration(delay)
        else:
            delay = self.command_delay
            # 后续动作以前一个已排程动作的时间为基准，避免连续按键在
            # 同一时刻到期后被 publish() 一次性全部执行。
            base_time = max(now, self.last_scheduled_time)
            due_time = base_time + rospy.Duration(delay)
        self.pending_commands.append((due_time, action))
        self.has_scheduled_command = True
        self.last_scheduled_time = due_time
        rospy.loginfo(
            "keyboard %s 已排队，将在 %.2f 秒后执行（届时按实际位姿计算 target）；队列长度=%d",
            action, delay, len(self.pending_commands))

    def publish(self, _event):
        now = rospy.Time.now()
        # 到达动作的计划时间后，成对发布目标位姿和动作标签。
        while self.pending_commands and self.pending_commands[0][0] <= now:
            _, action = self.pending_commands.popleft()
            current = self.current_pose()
            if current is None:
                rospy.logwarn("执行动作 %s 时没有有效实际位姿，等待下一次重试", action)
                self.pending_commands.appendleft((now + rospy.Duration(0.1), action))
                break
            target = self.target_for_action(action, current)
            self.target = target
            self.target.header.stamp = now
            self.reference_pub.publish(self.target)
            self.action_pub.publish(String(data=action))
            rospy.loginfo(
                "执行延迟键盘动作：%s，target=(x=%.3f,y=%.3f,z=%.3f,yaw=%.3f)，剩余队列=%d",
                action, self.target.pose.position.x, self.target.pose.position.y,
                self.target.pose.position.z,
                yaw_from_quat(self.target.pose.orientation),
                len(self.pending_commands))
        if self.target is None:
            return
        self.target.header.stamp = now
        self.reference_pub.publish(self.target)

    def shutdown(self):
        if self.target is not None:
            self.target.header.stamp = rospy.Time.now()
            self.reference_pub.publish(self.target)


def main():
    rospy.init_node("keyboard_reference_publisher")
    node = KeyboardReferencePublisher()
    if not sys.stdin.isatty():
        rospy.logfatal("keyboard_reference_publisher requires an interactive terminal")
        return

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        rospy.loginfo("Keyboard control: W=forward, A/D=turn, SPACE=STOP, Q=quit; 延迟由 command_delay_s 控制")
        while not rospy.is_shutdown():
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                break
            node.process_key(key)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
