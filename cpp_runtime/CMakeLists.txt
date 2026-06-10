cmake_minimum_required(VERSION 3.10)

project(pipeline_geometry_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pipeline_geometry_runtime
    src/main.cpp
    src/pipeline_geometry.cpp
)

target_include_directories(pipeline_geometry_runtime PRIVATE
    include
)
