#!/usr/bin/env python3

import copy
import threading
import time

import rospy
from geometry_msgs.msg import Twist, TwistStamped


class TwistConverter:
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "base")
        self.input_topic = rospy.get_param("~input_topic", "/twist_mux/cmd_vel")
        self.output_topic = rospy.get_param("~output_topic", "/motion_reference/command_twist")
        self.publish_rate = rospy.get_param("~publish_rate", 20.0)
        self.command_timeout = rospy.get_param("~command_timeout", 0.5)

        if self.publish_rate <= 0.0:
            raise ValueError("~publish_rate must be > 0.0")

        self.latest_twist = None
        self.last_twist_wall_time = None
        self.command_was_active = False
        self.latest_twist_lock = threading.Lock()

        self.pub = rospy.Publisher(
            self.output_topic,
            TwistStamped,
            queue_size=1
        )

        self.sub = rospy.Subscriber(
            self.input_topic,
            Twist,
            self.callback,
            queue_size=1
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.publish_latest_twist
        )

        rospy.loginfo(
            "Twist converter: %s -> %s at %.1f Hz, timeout %.2f s",
            self.input_topic,
            self.output_topic,
            self.publish_rate,
            self.command_timeout
        )

    def callback(self, msg):
        with self.latest_twist_lock:
            self.latest_twist = copy.deepcopy(msg)
            self.last_twist_wall_time = time.monotonic()
            self.command_was_active = True

    def publish_latest_twist(self, event):
        now = rospy.Time.now()

        with self.latest_twist_lock:
            command_is_fresh = (
                self.latest_twist is not None
                and self.last_twist_wall_time is not None
                and (time.monotonic() - self.last_twist_wall_time) <= self.command_timeout
            )

            if not command_is_fresh:
                if self.command_was_active:
                    self.command_was_active = False
                    twist = Twist()
                else:
                    return

            else:
                twist = copy.deepcopy(self.latest_twist)

        out = TwistStamped()
        out.header.stamp = now
        out.header.frame_id = self.frame_id
        out.twist = twist

        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("twist_converter")
    node = TwistConverter()
    rospy.spin()
