FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV ROS_VERSION=2

RUN apt-get update && apt-get install -y \
    bash-completion \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    libgl1 \
    libgtk-3-0 \
    libxkbcommon-x11-0 \
    mesa-utils \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-venv \
    ros-jazzy-cv-bridge \
    ros-jazzy-ros-gz \
    ros-jazzy-rqt-image-view \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-ur \
    ros-jazzy-xacro \
    python3-numpy \
    python3-opencv \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/venvs/ros && \
    /opt/venvs/ros/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venvs/ros/bin/python -m pip install --no-cache-dir ultralytics

RUN rosdep init || true

WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src && \
    git clone --depth 1 --branch jazzy \
    https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation.git \
    /ros2_ws/src/ur_simulation_gz

RUN bash -lc "source /opt/ros/jazzy/setup.bash && \
    source /opt/venvs/ros/bin/activate && \
    cd /ros2_ws && \
    colcon build --symlink-install && \
    echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc && \
    echo 'source /opt/venvs/ros/bin/activate' >> /etc/bash.bashrc && \
    echo 'source /ros2_ws/install/setup.bash' >> /etc/bash.bashrc"

# Include this package in standalone image builds. The compose bind mount still
# supports rapid source iteration during development.
COPY . /ros2_ws/src/ur_vision_avoidance
RUN bash -lc "source /opt/ros/jazzy/setup.bash && \
    source /opt/venvs/ros/bin/activate && \
    cd /ros2_ws && \
    colcon build --symlink-install"

CMD ["bash", "-lc", "source /opt/ros/jazzy/setup.bash && source /opt/venvs/ros/bin/activate && exec bash"]
