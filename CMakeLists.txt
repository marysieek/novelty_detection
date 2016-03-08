cmake_minimum_required(VERSION 3.3)
project(novelty_detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build) 

add_subdirectory(src)
