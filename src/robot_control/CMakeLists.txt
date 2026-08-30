cmake_minimum_required(VERSION 3.0.2)
project(robot_control)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  std_msgs
  visualization_msgs
)

add_message_files(
  FILES
  ActionRequest.msg
  ActionStatus.msg
)

generate_messages(DEPENDENCIES geometry_msgs)

catkin_package(
  CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs rospy std_msgs visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/cascaded_pid_tracker.py
  scripts/action_executor.py
  scripts/reference_tuning_publisher.py
  scripts/keyboard_reference_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
