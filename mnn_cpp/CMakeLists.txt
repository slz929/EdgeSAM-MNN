cmake_minimum_required(VERSION 3.0)
project(sam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)

# include dir
# include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)
include_directories(/home/local/file/MNN/include/)
include_directories(/home/local/file/MNN/tools/cv/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)
# link_directories(/home/local/file/MNN/build/libMNN.so)
# link_directories(/home/local/file/MNN/build/express/libMNN_Express.so)
# link_directories(/home/local/file/MNN/build/MNN/build/tools/cv/libMNNOpenCV.so)

message("mnn msg " ${MNN})
# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/sam_demo.cpp)


# target
add_executable(sam_demo ${SRCS})

target_link_libraries(sam_demo MNN MNN_Express MNNOpenCV ${OpenCV_LIBS})


