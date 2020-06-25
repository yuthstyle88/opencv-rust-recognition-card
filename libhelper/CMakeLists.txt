# Standard CMake prelude - nothing special
cmake_minimum_required(VERSION 3.0)
project(LibHelper CXX)
find_package( OpenCV REQUIRED )
add_library(libhelper SHARED cv_services.cpp)
target_link_libraries( libhelper ${OpenCV_LIBS} )
install(TARGETS libhelper DESTINATION .)
