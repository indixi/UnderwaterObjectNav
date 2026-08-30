#!/usr/bin/env python3
"""Publish a point position reference and velocity references for PID tuning."""
import math

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Path


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


class ReferenceTuningPublisher:
    def __init__(self):
        root = "~reference_tuning_publisher/"
        self.frame_id = rospy.get_param(root + "frame_id", "world_ned")
        self.velocity_frame_id = rospy.get_param(root + "velocity_frame_id", self.frame_id)
        self.pose_topic = rospy.get_param(root + "pose_topic", "/aquaflow/nominal_pose")
        self.velocity_topic = rospy.get_param(root + "velocity_topic", "/robot_control/reference_velocity")
        self.rate_hz = max(1.0, float(rospy.get_param(root + "publish_rate_hz", 10.0)))
        self.trajectory_type = rospy.get_param(root + "trajectory/type", "line")
        self.spacing = max(0.05, float(rospy.get_param(root + "trajectory/spacing", 0.25))) # spacing between points in the path，最小是0.05
        self.x_start = float(rospy.get_param(root + "trajectory/x_start", -5.0))
        self.x_end = float(rospy.get_param(root + "trajectory/x_end", 5.0))
        self.y = float(rospy.get_param(root + "trajectory/y", 0.0))
        self.z = float(rospy.get_param(root + "trajectory/z", 0.0))
        self.yaw = float(rospy.get_param(root + "trajectory/yaw", 0.0))
        self.eight_cx = float(rospy.get_param(root + "trajectory/figure_eight_center_x", 0.0))
        self.eight_cy = float(rospy.get_param(root + "trajectory/figure_eight_center_y", 0.0))
        self.eight_ax = float(rospy.get_param(root + "trajectory/figure_eight_amplitude_x", 2.0))
        self.eight_ay = float(rospy.get_param(root + "trajectory/figure_eight_amplitude_y", 1.5))
        self.eight_points = max(20, int(rospy.get_param(root + "trajectory/figure_eight_points", 160)))
        self.velocity_type = rospy.get_param(root + "velocity_profile/type", "constant")
        self.vx = float(rospy.get_param(root + "velocity_profile/vx", 0.10))
        self.vy = float(rospy.get_param(root + "velocity_profile/vy", 0.0))
        self.vz = float(rospy.get_param(root + "velocity_profile/vz", 0.0))
        self.yaw_rate = float(rospy.get_param(root + "velocity_profile/yaw_rate", 0.0))
        self.sine_amplitude = float(rospy.get_param(root + "velocity_profile/sine_amplitude", 0.10))
        self.sine_frequency = float(rospy.get_param(root + "velocity_profile/sine_frequency_hz", 0.10))
        self.step_time = float(rospy.get_param(root + "velocity_profile/step_time_s", 5.0))

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.velocity_pub = rospy.Publisher(self.velocity_topic, TwistStamped, queue_size=10)
        self.start_time = rospy.Time.now()
        self.reference_pose = self.build_path().poses[0]
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.publish)
        rospy.on_shutdown(self.shutdown)

    def build_path(self):
        path = Path()
        path.header.frame_id = self.frame_id
        if self.trajectory_type == "figure_eight":
            count = self.eight_points
        elif self.trajectory_type == "hold":
            count = 1
        elif self.trajectory_type == "line":
            count = max(2, int(math.ceil(abs(self.x_end - self.x_start) / self.spacing)) + 1)
        else:
            raise rospy.ROSInitException("unsupported trajectory type: %s" % self.trajectory_type)

        for i in range(count):
            u = 0.0 if count == 1 else float(i) / float(count - 1)
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            if self.trajectory_type == "figure_eight":
                theta = -0.5 * math.pi + 2.0 * math.pi * u
                st, ct = math.sin(theta), math.cos(theta)
                pose.pose.position.x = self.eight_cx + self.eight_ax * st
                pose.pose.position.y = self.eight_cy + self.eight_ay * st * ct
                dx = self.eight_ax * ct
                dy = self.eight_ay * (ct * ct - st * st)
                pose_yaw = math.atan2(dy, dx)
            else:
                pose.pose.position.x = self.x_start if self.trajectory_type == "hold" else self.x_start + u * (self.x_end - self.x_start)
                pose.pose.position.y = self.y
                pose_yaw = self.yaw
            pose.pose.position.z = self.z
            q = quat_from_yaw(pose_yaw)
            pose.pose.orientation.x, pose.pose.orientation.y = q[0], q[1]
            pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]
            path.poses.append(pose)
        return path

    def velocity_values(self, elapsed):
        if self.velocity_type == "sine":
            value = self.sine_amplitude * math.sin(2.0 * math.pi * self.sine_frequency * elapsed)
            return value, self.vy, self.vz, self.yaw_rate
        if self.velocity_type == "step":
            scale = 0.0 if elapsed < self.step_time else 1.0
            return scale * self.vx, scale * self.vy, scale * self.vz, scale * self.yaw_rate
        if self.velocity_type != "constant":
            rospy.logwarn_throttle(5.0, "unsupported velocity profile: %s", self.velocity_type)
        return self.vx, self.vy, self.vz, self.yaw_rate

    def publish(self, _event):
        stamp = rospy.Time.now()
        self.reference_pose.header.stamp = stamp
        self.pose_pub.publish(self.reference_pose)
        elapsed = (stamp - self.start_time).to_sec()
        vx, vy, vz, yaw_rate = self.velocity_values(max(0.0, elapsed))
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.velocity_frame_id
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vx, vy, vz
        msg.twist.angular.z = yaw_rate
        self.velocity_pub.publish(msg)

    def shutdown(self):
        self.velocity_pub.publish(TwistStamped())


if __name__ == "__main__":
    rospy.init_node("reference_tuning_publisher")
    ReferenceTuningPublisher()
    rospy.spin()

