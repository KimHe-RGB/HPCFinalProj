cmake_minimum_required(VERSION 3.25)
project(parareal)

set(CMAKE_CXX_STANDARD 17)
include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/)

#First Executable
add_executable(baseline baseline.cpp)
#target_link_libraries(baseline PRIVATE Eigen3::Eigen)

#Second Executable
add_executable(simple simple.cpp)
#target_link_libraries(simple PRIVATE Eigen3::Eigen)


