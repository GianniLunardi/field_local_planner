#!/usr/bin/env python3

import rospy
import rosservice


class SlamMapPublisher:
    def __init__(self):
        self.localization_service = rospy.get_param(
            "~localization_service",
            "/slam/toggle_localization"
        )
        self.mapping_service = rospy.get_param(
            "~mapping_service",
            "/slam/toggle_mapping"
        )
        self.publish_map_service = rospy.get_param(
            "~publish_map_service",
            "/slam/publish_map"
        )
        self.publish_period = rospy.get_param("~publish_period", 3.0)

        if self.publish_period <= 0.0:
            raise ValueError("~publish_period must be > 0.0")

        self.localization_service_class = self.wait_for_service_class(
            self.localization_service
        )
        self.mapping_service_class = self.wait_for_service_class(
            self.mapping_service
        )
        self.publish_map_service_class = self.wait_for_service_class(
            self.publish_map_service
        )

        self.enable_localization = rospy.ServiceProxy(
            self.localization_service,
            self.localization_service_class
        )
        self.enable_mapping = rospy.ServiceProxy(
            self.mapping_service,
            self.mapping_service_class
        )
        self.publish_map = rospy.ServiceProxy(
            self.publish_map_service,
            self.publish_map_service_class
        )

        self.enable_slam_modes()

        self.timer = rospy.Timer(
            rospy.Duration(self.publish_period),
            self.publish_map_callback
        )

        rospy.loginfo(
            "SLAM map publisher started: publishing map every %.2f s",
            self.publish_period
        )

    def wait_for_service_class(self, service):
        rospy.loginfo("Waiting for service [%s]", service)
        rospy.wait_for_service(service)
        return rosservice.get_service_class_by_name(service)

    def enable_slam_modes(self):
        localization_response = self.call_toggle_service(
            self.enable_localization,
            self.localization_service_class,
            True
        )
        rospy.loginfo("Enabled SLAM localization: %s", localization_response)

        mapping_response = self.call_toggle_service(
            self.enable_mapping,
            self.mapping_service_class,
            True
        )
        rospy.loginfo("Enabled SLAM mapping: %s", mapping_response)

    def call_toggle_service(self, proxy, service_class, enable):
        request = service_class._request_class()

        if hasattr(request, "enable"):
            request.enable = enable
        elif hasattr(request, "data"):
            request.data = enable
        else:
            raise AttributeError(
                "Toggle service request must expose 'enable' or 'data'"
            )

        return proxy(request)

    def publish_map_callback(self, event):
        try:
            request = self.publish_map_service_class._request_class()
            self.publish_map(request)
            rospy.logdebug("Requested SLAM map publication")
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed to call [%s]: %s", self.publish_map_service, exc)


if __name__ == "__main__":
    rospy.init_node("slam_map_publisher")
    node = SlamMapPublisher()
    rospy.spin()
