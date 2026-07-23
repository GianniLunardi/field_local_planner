#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-noetic}"
PLANNER_WS="${PLANNER_WS:-/planner_ws}"
PLANNER_LAUNCH_PACKAGE="${PLANNER_LAUNCH_PACKAGE:-field_local_planner_ros}"
PLANNER_LAUNCH_FILE="${PLANNER_LAUNCH_FILE:-rmp.launch}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CATKIN_EXTEND_PATH="${CATKIN_EXTEND_PATH:-/opt/ros/${ROS_DISTRO}}"

source "/opt/ros/${ROS_DISTRO}/setup.bash"

MODE="${1:-run}"
shift || true

case "${MODE}" in
  build)
    cd "${PLANNER_WS}"
    catkin config --extend "${CATKIN_EXTEND_PATH}" --cmake-args "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    exec catkin build field_local_planner_ros field_local_planner_rmp_plugin "$@"
    ;;
  run)
    mkdir -p "${PLANNER_WS}/bags"
    source "${PLANNER_WS}/devel/setup.bash"
    exec roslaunch "${PLANNER_LAUNCH_PACKAGE}" "${PLANNER_LAUNCH_FILE}" "$@"
    ;;
  viz)
    mkdir -p "${PLANNER_WS}/bags"
    source "${PLANNER_WS}/devel/setup.bash"
    exec roslaunch "${PLANNER_LAUNCH_PACKAGE}" rviz.launch "$@"
    ;;
  *)
    echo "Usage: $0 {build|run|viz}" >&2
    exit 1
    ;;
esac
