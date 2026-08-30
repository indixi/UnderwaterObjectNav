#!/usr/bin/env python3
"""Keyboard action-request publisher.

This node only translates keyboard input into numbered action requests. The
action_executor node owns timing, target generation, and completion handling.
"""
import select
import sys
import termios
import tty

import rospy

from robot_control.msg import ActionRequest


ACTIONS = {"w": "FORWARD", "a": "TURN_LEFT", "d": "TURN_RIGHT",
           " ": "STOP"}


class KeyboardActionPublisher:
    def __init__(self):
        root = "~keyboard_reference_publisher/"
        self.request_topic = rospy.get_param(
            root + "request_topic", "/underwater_objectnav/action_request")
        self.key_repeat_guard = max(
            0.0, float(rospy.get_param(root + "key_repeat_guard_s", 0.10)))
        self.last_key_time = rospy.Time(0)
        self.next_action_id = 1
        self.request_pub = rospy.Publisher(
            self.request_topic, ActionRequest, queue_size=20)

    def process_key(self, key):
        if key not in ACTIONS:
            return
        now = rospy.Time.now()
        if (now - self.last_key_time).to_sec() < self.key_repeat_guard:
            return
        self.last_key_time = now

        msg = ActionRequest()
        msg.action_id = self.next_action_id
        msg.action = ACTIONS[key]
        msg.request_time = now
        self.next_action_id += 1
        self.request_pub.publish(msg)
        rospy.loginfo("keyboard request id=%d action=%s",
                      msg.action_id, msg.action)


def main():
    rospy.init_node("keyboard_reference_publisher")
    node = KeyboardActionPublisher()
    if not sys.stdin.isatty():
        rospy.logfatal("keyboard_reference_publisher requires an interactive terminal")
        return

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        rospy.loginfo("Keyboard control: W=forward, A/D=turn, SPACE=STOP, Q=quit")
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
