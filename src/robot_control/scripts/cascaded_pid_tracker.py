#!/usr/bin/env python3
"""BricsBot position--velocity cascaded PID controller.

The node intentionally preserves planar_pid_tracker's ROS interface:
PoseStamped + Odometry -> WrenchStamped -> brics6_thruster_allocator.
"""
import math

import rospy
from geometry_msgs.msg import Point, PoseStamped, TwistStamped, WrenchStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class CascadedPIDTracker:
    def __init__(self):
        root = "~cascaded_pid_tracker/"
        self.vehicle_name = rospy.get_param(
            "~vehicle_name", rospy.get_param(root + "vehicle_name", "bricsbot"))
        self.rate_hz = max(1.0, float(rospy.get_param(root + "rate_hz", 50.0)))
        self.reference_topic = rospy.get_param(             #这里是嵌套用法
            "~reference_topic", rospy.get_param(
                root + "reference/topic", "/aquaflow/nominal_pose"))
        self.reference_mode = rospy.get_param(
            "~reference_mode", rospy.get_param(root + "reference/mode", "position"))
        self.velocity_reference_topic = rospy.get_param(
            "~velocity_reference_topic", rospy.get_param(
                root + "reference/velocity_topic", "/robot_control/reference_velocity"))
        self.reference_frame = rospy.get_param(
            root + "reference/frame_id", "world_ned")


        #读取ROS参数服务器中的PID控制器参数，如果没有设置则使用默认值
        self.kp_pos_xy = self._pair(root + "position_loop/kp_xy", [0.6, 0.6])
        self.ki_pos_xy = self._pair(root + "position_loop/ki_xy", [0.0, 0.0])
        self.kp_pos_z = self._scalar(root + "position_loop/kp_z", 0.6)
        self.ki_pos_z = self._scalar(root + "position_loop/ki_z", 0.0)
        self.kp_pos_yaw = self._scalar(root + "position_loop/kp_yaw", 0.8)
        self.ki_pos_yaw = self._scalar(root + "position_loop/ki_yaw", 0.0)
        self.pos_i_lim_xy = self._pair(root + "position_loop/integral_limit_xy", [1.0, 1.0])
        self.pos_i_lim_z = self._scalar(root + "position_loop/integral_limit_z", 1.0)
        self.pos_i_lim_yaw = self._scalar(root + "position_loop/integral_limit_yaw", 1.0)
        self.max_vel_xy = self._pair(root + "position_loop/max_velocity_xy", [0.25, 0.25])
        self.max_vel_z = self._scalar(root + "position_loop/max_velocity_z", 0.10)
        self.max_yaw_rate = self._scalar(root + "position_loop/max_yaw_rate", 0.25)

        self.kp_vel_xy = self._pair(root + "velocity_loop/kp_xy", [8.0, 8.0])
        self.ki_vel_xy = self._pair(root + "velocity_loop/ki_xy", [0.0, 0.0])
        self.kd_vel_xy = self._pair(root + "velocity_loop/kd_xy", [0.5, 0.5])
        self.kp_vel_z = self._scalar(root + "velocity_loop/kp_z", 8.0)
        self.ki_vel_z = self._scalar(root + "velocity_loop/ki_z", 0.0)
        self.kd_vel_z = self._scalar(root + "velocity_loop/kd_z", 0.5)
        self.kp_vel_yaw = self._scalar(root + "velocity_loop/kp_yaw", 1.5)
        self.ki_vel_yaw = self._scalar(root + "velocity_loop/ki_yaw", 0.0)
        self.kd_vel_yaw = self._scalar(root + "velocity_loop/kd_yaw", 0.05)
        self.vel_i_lim_xy = self._pair(root + "velocity_loop/integral_limit_xy", [1.0, 1.0])
        self.vel_i_lim_z = self._scalar(root + "velocity_loop/integral_limit_z", 1.0)
        self.vel_i_lim_yaw = self._scalar(root + "velocity_loop/integral_limit_yaw", 1.0)
        self.max_force_xy = self._pair(root + "velocity_loop/max_force_xy", [4.0, 4.0])
        self.max_force_z = self._scalar(root + "velocity_loop/max_force_z", 20.0)
        self.max_torque_z = self._scalar(root + "velocity_loop/max_torque_z", 1.4)
        cutoff = self._scalar(root + "velocity_loop/derivative_filter_hz", 8.0) #这是截止频率，单位是赫兹，默认值是8.0
        #Previous-result coefficient of a first-order RC low-pass filter:
        # alpha = tau / (tau + dt) = 1 / (1 + 2*pi*fc*dt).
        self.derivative_alpha = 1.0 / (1.0 + 2.0 * math.pi * max(0.1, cutoff) / self.rate_hz)

        self.odom_timeout = self._scalar(root + "safety/odom_timeout_s", 0.25)
        self.reference_timeout = self._scalar(root + "safety/reference_timeout_s", 0.50)
        self.max_state_age = self._scalar(root + "safety/max_state_age_s", 0.25)
        self.future_tolerance = self._scalar(root + "safety/future_stamp_tolerance_s", 0.02)
        self.dt_min = self._scalar(root + "safety/dt_min_s", 0.001)
        self.dt_max = self._scalar(root + "safety/dt_max_s", 0.20)

        self.odom = None
        self.reference_pose = None
        self.velocity_reference = None
        self.last_control_time = None
        self.last_odom_stamp = None
        self.last_vel_error = [0.0, 0.0, 0.0, 0.0]
        self.last_vel_derivative = [0.0, 0.0, 0.0, 0.0]
        self.pos_i = [0.0, 0.0, 0.0, 0.0]
        self.vel_i = [0.0, 0.0, 0.0, 0.0]

        self.force_pub = rospy.Publisher("/controller/generalized_force",
                                         WrenchStamped, queue_size=1)
        self.error_pubs = [rospy.Publisher(topic, Float64, queue_size=10)
                           for topic in ("/aquaflow/tracking_error/x_body_m",
                                         "/aquaflow/tracking_error/y_body_m",
                                         "/aquaflow/tracking_error/yaw_rad",
                                         "/aquaflow/tracking_error/xy_norm_m")]
        self.vel_ref_pubs = [rospy.Publisher(topic, Float64, queue_size=10)
                             for topic in ("/aquaflow/cascade_pid/velocity_reference/x",
                                           "/aquaflow/cascade_pid/velocity_reference/y",
                                           "/aquaflow/cascade_pid/velocity_reference/z",
                                           "/aquaflow/cascade_pid/velocity_reference/yaw_rate")]
        self.vel_err_pubs = [rospy.Publisher(topic, Float64, queue_size=10)
                             for topic in ("/aquaflow/cascade_pid/velocity_error/x",
                                           "/aquaflow/cascade_pid/velocity_error/y",
                                           "/aquaflow/cascade_pid/velocity_error/z",
                                           "/aquaflow/cascade_pid/velocity_error/yaw_rate")]
        self.time_state_age_pub = rospy.Publisher(
            "/aquaflow/cascade_pid/time/state_age_s", Float64, queue_size=10)
        self.time_reference_age_pub = rospy.Publisher(
            "/aquaflow/cascade_pid/time/reference_age_s", Float64, queue_size=10)
        self.marker_pub = rospy.Publisher("/aquaflow/tracking_error_markers",
                                          MarkerArray, queue_size=1)
        self.target_marker_pub = rospy.Publisher("/aquaflow/tracking_target_markers",
                                                 MarkerArray, queue_size=1)

        rospy.Subscriber("/%s/odometry" % self.vehicle_name, Odometry,
                         self.odom_cb, queue_size=1)
        rospy.Subscriber(self.reference_topic, PoseStamped, self.reference_pose_cb, queue_size=1)
        if self.reference_mode == "velocity":
            rospy.Subscriber(self.velocity_reference_topic, TwistStamped,
                             self.velocity_reference_cb, queue_size=1)
        elif self.reference_mode != "position":
            raise rospy.ROSInitException(
                "reference/mode must be position or velocity, got %s" % self.reference_mode)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.update)
        rospy.on_shutdown(self.shutdown)

    @staticmethod
    def _scalar(name, default):
        return float(rospy.get_param(name, default))

    @staticmethod
    def _pair(name, default):
        value = rospy.get_param(name, default)
        return [float(value[0]), float(value[1])]

    def odom_cb(self, msg):
        self.odom = msg

    def reference_pose_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.reference_frame:
            rospy.logwarn_throttle(5.0, "reference frame is %s, expected %s",
                                   msg.header.frame_id, self.reference_frame)
        self.reference_pose = msg

    def velocity_reference_cb(self, msg):
        self.velocity_reference = msg

    def reset_integrators(self):
        self.pos_i = [0.0, 0.0, 0.0, 0.0]
        self.vel_i = [0.0, 0.0, 0.0, 0.0]
        self.last_vel_error = [0.0, 0.0, 0.0, 0.0]
        self.last_vel_derivative = [0.0, 0.0, 0.0, 0.0]

    def publish_zero(self, stamp=None):
        msg = WrenchStamped()
        msg.header.stamp = stamp or rospy.Time.now()
        msg.header.frame_id = "base_link"
        self.force_pub.publish(msg)

    @staticmethod
    def arrow(marker_id, frame, stamp, start, end, color, scale=0.035):
        marker = Marker()
        marker.header.frame_id = frame
        marker.header.stamp = stamp
        marker.ns, marker.id = "aquaflow_tracking_error", marker_id
        marker.type, marker.action = Marker.ARROW, Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = scale, 2.0 * scale, 3.0 * scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.points = [Point(*start), Point(*end)]
        return marker

    def publish_markers(self, now, p, target, yaw, target_yaw, ex, ey):
        z = p.z + 0.12
        current = (p.x, p.y, z)
        desired = (target.position.x, target.position.y, z)
        markers = MarkerArray()
        markers.markers.append(self.arrow(0, self.reference_frame, now, current, desired,
                                          (1.0, 0.85, 0.0, 0.95), 0.028))
        length = 0.42
        markers.markers.append(self.arrow(
            1, self.reference_frame, now, current,
            (p.x + length * math.cos(yaw), p.y + length * math.sin(yaw), z),
            (0.10, 0.55, 1.0, 0.95)))
        markers.markers.append(self.arrow(
            2, self.reference_frame, now, current,
            (p.x + length * math.cos(target_yaw), p.y + length * math.sin(target_yaw), z),
            (1.0, 0.35, 0.05, 0.95)))
        self.marker_pub.publish(markers)

    def update(self, _event):
        now = rospy.Time.now()
        # Check for valid odometry and reference data, and reset integrators if necessary
        if now == rospy.Time():
            self.publish_zero(now)
            return
        if self.odom is None:
            self.reset_integrators()
            self.publish_zero(now)
            return

        odom_stamp = self.odom.header.stamp
        #经过位置外环
        if self.reference_mode == "position":
            if self.reference_pose is None:
                self.reset_integrators()
                self.publish_zero(now)
                return
            reference_stamp = self.reference_pose.header.stamp
        #不经过位置外环，直接使用速度参考
        else:
            if self.velocity_reference is None:
                self.reset_integrators()
                self.publish_zero(now)
                return
            reference_stamp = self.velocity_reference.header.stamp
        #检查速度参考的时间戳是否过期
        state_age = (now - odom_stamp).to_sec()
        reference_age = (now - reference_stamp).to_sec()
        self.time_state_age_pub.publish(Float64(data=state_age))
        self.time_reference_age_pub.publish(Float64(data=reference_age))
        if (odom_stamp == rospy.Time() or reference_stamp == rospy.Time() or
                state_age < -self.future_tolerance or
                reference_age < -self.future_tolerance or
                state_age > self.max_state_age or
                state_age > self.odom_timeout or
                reference_age > self.reference_timeout):
            self.reset_integrators()
            self.publish_zero(now)
            return

        if self.last_control_time is None:
            dt = 1.0 / self.rate_hz
        else:
            raw_dt = (now - self.last_control_time).to_sec()
            if raw_dt <= 0.0 or raw_dt > self.dt_max:
                self.reset_integrators()
            dt = clamp(raw_dt, self.dt_min, self.dt_max)
        self.last_control_time = now

        p = self.odom.pose.pose.position
        yaw = yaw_from_quat(self.odom.pose.pose.orientation)
        if self.reference_mode == "position":
            target = self.reference_pose.pose
            dx, dy = target.position.x - p.x, target.position.y - p.y
            c, s = math.cos(yaw), math.sin(yaw)
            ex, ey = c * dx + s * dy, -s * dx + c * dy
            ez = target.position.z - p.z
            eyaw = wrap(yaw_from_quat(target.orientation) - yaw)
            self.pos_i[0] = clamp(self.pos_i[0] + ex * dt, -self.pos_i_lim_xy[0], self.pos_i_lim_xy[0])
            self.pos_i[1] = clamp(self.pos_i[1] + ey * dt, -self.pos_i_lim_xy[1], self.pos_i_lim_xy[1])
            self.pos_i[2] = clamp(self.pos_i[2] + ez * dt, -self.pos_i_lim_z, self.pos_i_lim_z)
            self.pos_i[3] = clamp(self.pos_i[3] + eyaw * dt, -self.pos_i_lim_yaw, self.pos_i_lim_yaw)
            vref = [
                clamp(self.kp_pos_xy[0] * ex + self.ki_pos_xy[0] * self.pos_i[0], -self.max_vel_xy[0], self.max_vel_xy[0]),
                clamp(self.kp_pos_xy[1] * ey + self.ki_pos_xy[1] * self.pos_i[1], -self.max_vel_xy[1], self.max_vel_xy[1]),
                clamp(self.kp_pos_z * ez + self.ki_pos_z * self.pos_i[2], -self.max_vel_z, self.max_vel_z),
                clamp(self.kp_pos_yaw * eyaw + self.ki_pos_yaw * self.pos_i[3], -self.max_yaw_rate, self.max_yaw_rate),
            ]
        else:
            target = None
            ex = ey = ez = eyaw = 0.0
            msg = self.velocity_reference.twist
            vref = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z]

        twist = self.odom.twist.twist
        vactual = [twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z]
        verr = [vref[i] - vactual[i] for i in range(4)]
        for i in range(4):
            self.vel_i[i] += verr[i] * dt
        self.vel_i[0] = clamp(self.vel_i[0], -self.vel_i_lim_xy[0], self.vel_i_lim_xy[0])
        self.vel_i[1] = clamp(self.vel_i[1], -self.vel_i_lim_xy[1], self.vel_i_lim_xy[1])
        self.vel_i[2] = clamp(self.vel_i[2], -self.vel_i_lim_z, self.vel_i_lim_z)
        self.vel_i[3] = clamp(self.vel_i[3], -self.vel_i_lim_yaw, self.vel_i_lim_yaw)
        raw_derivative = [(verr[i] - self.last_vel_error[i]) / dt for i in range(4)]
        for i in range(4):
            self.last_vel_derivative[i] = (self.derivative_alpha * self.last_vel_derivative[i] +
                                           (1.0 - self.derivative_alpha) * raw_derivative[i])
        self.last_vel_error = verr
        wrench = [
            self.kp_vel_xy[0] * verr[0] + self.ki_vel_xy[0] * self.vel_i[0] + self.kd_vel_xy[0] * self.last_vel_derivative[0],
            self.kp_vel_xy[1] * verr[1] + self.ki_vel_xy[1] * self.vel_i[1] + self.kd_vel_xy[1] * self.last_vel_derivative[1],
            self.kp_vel_yaw * verr[3] + self.ki_vel_yaw * self.vel_i[3] + self.kd_vel_yaw * self.last_vel_derivative[3],
            self.kp_vel_z * verr[2] + self.ki_vel_z * self.vel_i[2] + self.kd_vel_z * self.last_vel_derivative[2],
        ]
        wrench[0] = clamp(wrench[0], -self.max_force_xy[0], self.max_force_xy[0])
        wrench[1] = clamp(wrench[1], -self.max_force_xy[1], self.max_force_xy[1])
        wrench[2] = clamp(wrench[2], -self.max_torque_z, self.max_torque_z)
        wrench[3] = clamp(wrench[3], -self.max_force_z, self.max_force_z)

        msg = WrenchStamped()
        msg.header.stamp = now
        msg.header.frame_id = "base_link"
        msg.wrench.force.x, msg.wrench.force.y = wrench[0], wrench[1]
        msg.wrench.torque.z, msg.wrench.force.z = wrench[2], wrench[3]
        self.force_pub.publish(msg)
        for pub, value in zip(self.error_pubs, (ex, ey, eyaw, math.hypot(ex, ey))):
            pub.publish(Float64(data=value))
        for pub, value in zip(self.vel_ref_pubs, vref):
            pub.publish(Float64(data=value))
        for pub, value in zip(self.vel_err_pubs, verr):
            pub.publish(Float64(data=value))
        if target is not None:
            self.publish_markers(now, p, target, yaw, yaw + eyaw, ex, ey)

    def shutdown(self):
        self.reset_integrators()
        self.publish_zero(rospy.Time.now())


if __name__ == "__main__":
    rospy.init_node("cascaded_pid_tracker")
    CascadedPIDTracker()
    rospy.spin()
