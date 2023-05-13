cmake_minimum_required(VERSION 3.25)
project(parareal)

set(CMAKE_CXX_STANDARD 17)
include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/)

# Add OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(parareal_naive PUBLIC OpenMP::OpenMP_CXX)
endif()

#First Executable
add_executable(error error.cpp heat_solver.h heat_solver.cpp parareal_naive.cpp)
#target_link_libraries(baseline PRIVATE Eigen3::Eigen)

#First Executable
add_executable(parareal_naive parareal_naive.cpp heat_solver.h heat_solver.cpp parareal_naive.cpp)
#target_link_libraries(baseline PRIVATE Eigen3::Eigen)
