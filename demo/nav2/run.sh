#!/bin/bash
set -eo pipefail
source /opt/ros/kilted/setup.bash
MAP=${MAP:-/venue/maps/default.yaml}
PARAMS=${PARAMS:-/anews-nav2/nav2_params.yaml}
ros2 run rmw_zenoh_cpp rmw_zenohd &
sleep 2
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:="$MAP" -p use_sim_time:=true &
ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_map \
  -p node_names:="['map_server']" -p autostart:=true -p use_sim_time:=true &
exec ros2 launch nav2_bringup navigation_launch.py params_file:="$PARAMS" use_sim_time:=True autostart:=True use_composition:=False
