cmake_minimum_required(VERSION 3.25)
project(parareal)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/)

#First Executable
add_executable(error error.cpp heat_solver.h heat_solver.cpp)
#target_link_libraries(baseline PRIVATE Eigen3::Eigen)

#First Executable
add_executable(parareal_naive parareal_naive.cpp heat_solver.h heat_solver.cpp)
find_package(OpenMP REQUIRED)
target_link_libraries(sample PRIVATE OpenMP::OpenMP_CXX)

# Add OpenMP
#find_package(OpenMP REQUIRED)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(parareal_naive PUBLIC OpenMP::OpenMP_CXX)
#endif()
