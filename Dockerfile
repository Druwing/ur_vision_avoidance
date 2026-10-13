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
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-ur \
    ros-jazzy-xacro \
    python3-numpy \
    python3-opencv \
    vim \
    ros-jazzy-message-filters \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /opt/venvs/ros && \
    /opt/venvs/ros/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venvs/ros/bin/python -m pip install --no-cache-dir "numpy==1.26.4" ultralytics
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

# ROS 2 console scripts are generated with #!/usr/bin/python3, so they do not
# see packages installed only in the venv (Ultralytics, stable-baselines3).
# Expose the venv site-packages to the system interpreter used by ros2 run/launch.
ENV PYTHONPATH=/opt/venvs/ros/lib/python3.12/site-packages
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Optional RL runtime (rl_policy_node). Keep NumPy 1.x for the ROS cv_bridge.
RUN /opt/venvs/ros/bin/python -m pip install --no-cache-dir \
    "numpy==1.26.4" "gymnasium>=0.29,<1.1" "stable-baselines3>=2.3,<3.0"

# Cache the Fuel chair used by worlds/obstacle_world.sdf. Containers are
# recreated on every run, and a failed download at launch ("Error in REST
# request") leaves the world unloaded.
RUN bash -lc "source /opt/ros/jazzy/setup.bash && \
    gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/Chair"

# Include this package in standalone image builds. The compose bind mount still
# supports rapid source iteration during development.
COPY . /ros2_ws/src/ur_vision_avoidance
RUN bash -lc "source /opt/ros/jazzy/setup.bash && \
    source /opt/venvs/ros/bin/activate && \
    cd /ros2_ws && \
    colcon build --symlink-install"

CMD ["bash", "-lc", "source /opt/ros/jazzy/setup.bash && source /opt/venvs/ros/bin/activate && exec bash"]
