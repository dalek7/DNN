project(hellolayer)
cmake_minimum_required(VERSION 2.8)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

add_executable(hellolayer main.cpp sin_layer.cpp)
target_link_libraries(hellolayer ${Caffe_LIBRARIES})




