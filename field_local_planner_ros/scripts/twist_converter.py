#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, TwistStamped

class TwistConverter:
    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "base")

        self.pub = rospy.Publisher(
            "/motion_reference/command_twist",
            TwistStamped,
            queue_size=10
        )

        self.sub = rospy.Subscriber(
            "/twist_mux/cmd_vel",
            Twist,
            self.callback
        )

        rospy.loginfo("Twist converter (twist to stamped)")

        rospy.spin()

    def callback(self, msg):
        out = TwistStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.frame_id
        out.twist = msg
        self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("twist_converter")
    node = TwistConverter()
