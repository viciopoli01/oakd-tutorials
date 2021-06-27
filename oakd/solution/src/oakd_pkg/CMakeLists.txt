cmake_minimum_required(VERSION 3.0.2)
project(oakd_pkg)

find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  geometry_msgs
  sensor_msgs
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
