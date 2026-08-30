cmake_minimum_required(VERSION 3.0.2)
project(data_generate)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs message_generation message_filters nav_msgs rospy
  sensor_msgs std_msgs cv_bridge
)

add_service_files(
  FILES
  StartEpisode.srv
  EndEpisode.srv
  CancelEpisode.srv
)

generate_messages(DEPENDENCIES geometry_msgs)

catkin_package(CATKIN_DEPENDS geometry_msgs message_runtime message_filters
  nav_msgs rospy sensor_msgs std_msgs cv_bridge)

catkin_install_python(PROGRAMS
  scripts/objectnav_data_collector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
