cmake_minimum_required(VERSION 3.10)
project(DRRGeneration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include libs)

# Automatically gather sources
file(GLOB SRC_FILES CONFIGURE_DEPENDS "src/*.cpp")

find_package(ZLIB REQUIRED)
add_executable(drr_gen ${SRC_FILES})
target_link_libraries(drr_gen ZLIB::ZLIB)
