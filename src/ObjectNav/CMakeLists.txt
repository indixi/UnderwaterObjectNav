cmake_minimum_required(VERSION 3.0.2)
project(objectnav)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_filters
  message_generation
  nav_msgs
  robot_control
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

add_message_files(
  FILES
  NavigationStatus.msg
)

add_service_files(
  FILES
  StartNavigation.srv
  CancelNavigation.srv
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS cv_bridge message_filters message_runtime nav_msgs
  robot_control rospy sensor_msgs std_msgs
)

catkin_install_python(PROGRAMS
  scripts/online_navigation_node.py
  scripts/inference_worker_launcher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY objectnav_bc/config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/objectnav_bc
)

install(DIRECTORY objectnav_bc/perception/model
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/objectnav_bc/perception
)

install(FILES
  README.md
  underwater_objectnav_algorithm_framework_gfl_bbox_depth_bc_mlp.md
  underwater_objectnav_bc_training_plan.md
  underwater_objectnav_gfl_depth_bc_reference.md
  underwater_objectnav_online_usage.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES objectnav_bc/README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/objectnav_bc
)
