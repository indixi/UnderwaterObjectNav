#!/usr/bin/env python3
"""Execute queued discrete actions independently of data collection.

The executor is the single owner of action ordering and completion.  It
publishes the current target for the PID controller and publishes STARTED /
SUCCEEDED status events for optional observers such as the data collector.
"""
import math
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from robot_control.msg import ActionRequest, ActionStatus


ACTIONS = ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ActionExecutor:
    def __init__(self):
        root = "~action_executor/"
        vehicle = rospy.get_param(root + "vehicle_name", "bricsbot")
        self.odom_topic = rospy.get_param(
            root + "odom_topic", "/%s/odometry" % vehicle)
        self.request_topic = rospy.get_param(
            root + "request_topic", "/underwater_objectnav/action_request")
        self.reference_topic = rospy.get_param(
            root + "reference_topic", "/aquaflow/nominal_pose")
        self.status_topic = rospy.get_param(
            root + "status_topic", "/underwater_objectnav/action_status")
        self.frame_id = rospy.get_param(root + "frame_id", "world_ned")
        self.rate_hz = max(1.0, float(rospy.get_param(root + "rate_hz", 20.0)))
        self.command_delay = max(
            0.0, float(rospy.get_param(root + "command_delay_s", 5.0)))
        self.initial_delay = max(
            0.0, float(rospy.get_param(root + "initial_command_delay_s", 0.0)))
        self.forward_distance = float(
            rospy.get_param(root + "forward_distance_m", 0.5))
        self.turn_angle = math.radians(float(
            rospy.get_param(root + "turn_angle_deg", 30.0)))
        self.odom_timeout = max(
            0.0, float(rospy.get_param(root + "odom_timeout_s", 0.25)))
        self.position_tolerance = float(
            rospy.get_param(root + "forward_position_tolerance_m", 0.035))
        self.yaw_tolerance = math.radians(float(
            rospy.get_param(root + "turn_angle_tolerance_deg", 3.0)))
        self.action_timeout = max(
            0.0, float(rospy.get_param(root + "action_timeout_s", 15.0)))

        self.odom = None
        self.pending = deque()
        self.last_scheduled_time = None
        self.current_request = None
        self.current_target = None
        self.start_time = None

        self.reference_pub = rospy.Publisher(
            self.reference_topic, PoseStamped, queue_size=1)
        self.status_pub = rospy.Publisher(
            self.status_topic, ActionStatus, queue_size=20)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(self.request_topic, ActionRequest,
                         self.request_cb, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.update)
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
            return None
        return self.odom.pose.pose

    def make_target(self, pose, action):
        x = pose.position.x
        y = pose.position.y
        z = pose.position.z
        yaw = yaw_from_quat(pose.orientation)
        target_yaw = yaw
        if action == "FORWARD":
            x += self.forward_distance * math.cos(yaw)
            y += self.forward_distance * math.sin(yaw)
        elif action == "TURN_LEFT":
            target_yaw += self.turn_angle
        elif action == "TURN_RIGHT":
            target_yaw -= self.turn_angle

        target = PoseStamped()
        target.header.frame_id = self.frame_id
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        q = quat_from_yaw(target_yaw)
        target.pose.orientation.x, target.pose.orientation.y = q[0], q[1]
        target.pose.orientation.z, target.pose.orientation.w = q[2], q[3]
        return target

    def request_cb(self, msg):
        action = msg.action.strip().upper()
        if action not in ACTIONS:
            rospy.logwarn("ignore unknown action %r", msg.action)
            return
        now = rospy.Time.now()
        if self.last_scheduled_time is None:
            due = now + rospy.Duration(self.initial_delay)
        else:
            due = max(now, self.last_scheduled_time) + \
                rospy.Duration(self.command_delay)
        self.pending.append((due, msg))
        self.last_scheduled_time = due
        rospy.loginfo("queued action id=%d action=%s due=%.3f queue=%d",
                      msg.action_id, action, due.to_sec(), len(self.pending))

    def publish_status(self, request, state, target=None,
                       start_time=None, finish_time=None, reason=""):
        msg = ActionStatus()
        msg.action_id = request.action_id
        msg.action = request.action.strip().upper()
        msg.state = state
        if target is not None:
            msg.target = target
        msg.start_time = start_time or rospy.Time()
        msg.finish_time = finish_time or rospy.Time()
        msg.failure_reason = reason
        self.status_pub.publish(msg)

    def start_next(self, now):
        if self.current_request is not None:
            return
        if not self.pending or self.pending[0][0] > now:
            return
        due, request = self.pending.popleft()
        pose = self.current_pose()
        if pose is None:
            self.pending.appendleft((now + rospy.Duration(0.1), request))
            return

        action = request.action.strip().upper()
        target = self.make_target(pose, action)
        start = rospy.Time.now()
        self.current_target = target
        self.start_time = start
        self.reference_pub.publish(target)
        self.publish_status(request, "STARTED", target, start_time=start)
        rospy.loginfo(
            "started action id=%d action=%s target=(%.3f, %.3f, %.3f, %.3f)",
            request.action_id, action, target.pose.position.x,
            target.pose.position.y, target.pose.position.z,
            yaw_from_quat(target.pose.orientation))

        if action == "STOP":
            self.publish_status(request, "STOPPED", target,
                                start_time=start, finish_time=rospy.Time.now())
            self.current_target = None
            self.start_time = None
            return
        self.current_request = request

    def finish_current(self, state, reason=""):
        request = self.current_request
        now = rospy.Time.now()
        self.publish_status(request, state, self.current_target,
                            start_time=self.start_time, finish_time=now,
                            reason=reason)
        rospy.loginfo("finished action id=%d state=%s",
                      request.action_id, state)
        self.current_request = None
        self.current_target = None
        self.start_time = None

    def update(self, _event):
        now = rospy.Time.now()
        if self.current_request is None:
            self.start_next(now)
        else:
            pose = self.current_pose()
            if pose is not None:
                target = self.current_target.pose
                dx = target.position.x - pose.position.x
                dy = target.position.y - pose.position.y
                distance = math.hypot(dx, dy)
                yaw_error = abs(wrap(
                    yaw_from_quat(target.orientation) -
                    yaw_from_quat(pose.orientation)))
                action = self.current_request.action.strip().upper()
                done = (distance <= self.position_tolerance
                        if action == "FORWARD"
                        else yaw_error <= self.yaw_tolerance)
                if done:
                    self.finish_current("SUCCEEDED")
                elif self.action_timeout > 0.0 and \
                        (now - self.start_time).to_sec() > self.action_timeout:
                    self.finish_current("TIMEOUT", "action_timeout")

        if self.current_target is not None:
            self.current_target.header.stamp = now
            self.reference_pub.publish(self.current_target)

    def shutdown(self):
        if self.current_target is not None:
            self.current_target.header.stamp = rospy.Time.now()
            self.reference_pub.publish(self.current_target)


if __name__ == "__main__":
    rospy.init_node("action_executor")
    ActionExecutor()
    rospy.spin()
