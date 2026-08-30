#!/usr/bin/env python3
"""Keyboard point-reference publisher for position-mode PID control.

W: create a target 0.5 m ahead of the current vehicle pose.
A/D: create a target with +/-30 deg yaw at the current position.
Space: hold the current pose.

The node publishes PoseStamped references; it does not publish PWM or force
commands directly.  Each key action is computed from the latest odometry.
"""
import math
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


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

        self.odom = None
        self.target = None
        self.last_key_time = rospy.Time(0)
        self.reference_pub = rospy.Publisher(
            self.reference_topic, PoseStamped, queue_size=1)
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

    def process_key(self, key):
        if key not in ("w", "a", "d", " "):
            return
        now = rospy.Time.now()
        if (now - self.last_key_time).to_sec() < self.key_repeat_guard:
            return
        current = self.current_pose()
        if current is None:
            return

        x = current.position.x
        y = current.position.y
        z = current.position.z
        yaw = yaw_from_quat(current.orientation)
        target_yaw = yaw
        target_x, target_y = x, y

        if key == "w":
            target_x += self.forward_distance * math.cos(yaw)
            target_y += self.forward_distance * math.sin(yaw)
        elif key == "a":
            target_yaw += self.turn_angle
        elif key == "d":
            target_yaw -= self.turn_angle

        self.target = self.make_target(target_x, target_y, z, target_yaw)
        self.last_key_time = now
        action = {"w": "forward", "a": "turn left", "d": "turn right",
                  " ": "hold"}[key]
        rospy.loginfo("keyboard %s: target=(%.3f, %.3f, %.3f), yaw=%.1f deg",
                      action, target_x, target_y, z, math.degrees(target_yaw))

    def publish(self, _event):
        if self.target is None:
            return
        self.target.header.stamp = rospy.Time.now()
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
        rospy.loginfo("Keyboard control: W=forward, A/D=turn, SPACE=hold, Q=quit")
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
