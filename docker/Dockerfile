ARG ROS_DISTRO=noetic
FROM osrf/ros:${ROS_DISTRO}-desktop-full

LABEL maintainer="Gianni.Lunardi@unibz.it"

# No user interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome
ENV ROS_DISTRO=${ROS_DISTRO}
SHELL ["/bin/bash", "-c"]

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install build tools, library dependencies and ROS packages used by the planners.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gnupg \
        libblosc-dev \
        libopencv-dev \
        libpcl-dev \
        libyaml-cpp-dev \
        pybind11-dev \
        python3 \
        python3-catkin-tools \
        python3-pip \
        python3-rosdep \
        ros-${ROS_DISTRO}-eigen-conversions \
        ros-${ROS_DISTRO}-filters \
        ros-${ROS_DISTRO}-gtsam \
        ros-${ROS_DISTRO}-grid-map-filters \
        ros-${ROS_DISTRO}-grid-map-msgs \
        ros-${ROS_DISTRO}-grid-map-ros \
        ros-${ROS_DISTRO}-grid-map-rviz-plugin \
        ros-${ROS_DISTRO}-grid-map-visualization \
        ros-${ROS_DISTRO}-move-base-msgs \
        ros-${ROS_DISTRO}-pcl-conversions \
        ros-${ROS_DISTRO}-pcl-ros \
        ros-${ROS_DISTRO}-teleop-twist-joy \
        ros-${ROS_DISTRO}-tf2-eigen \
        ros-${ROS_DISTRO}-twist-mux \
    && (rosdep init || true) \
    && (rosdep update --include-eol-distros || rosdep update)

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

RUN mkdir -p /planner_ws/src \
    && chown -R ${USERNAME}:${USERNAME} /planner_ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ${USERNAME}
WORKDIR /planner_ws

ENTRYPOINT ["/entrypoint.sh"]
CMD []
