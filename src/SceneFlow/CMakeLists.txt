cmake_minimum_required(VERSION 2.8)
project(SceneFlow)

IF(NOT CMAKE_BUILD_TYPE)
  #SET(CMAKE_BUILD_TYPE Release)
   SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -w -Wunused-parameter -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -w  -Wunused-parameter -O3 -march=native")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -W -Wunused-parameter -g -O0 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wunused-parameter -g -O0 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHCi11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


set(OpenCV_DIR /usr/local/opencv345/share/OpenCV)
message("OpenCV_DIR: " ${OpenCV_DIR})
message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
find_package(OpenCV 3 QUIET)

###########
## Build ##
###########

include_directories(
   ${OpenCV_INCLUDE_DIRS}
   ${PROJECT_SOURCE_DIR}
   ${PROJECT_SOURCE_DIR}/include/
   /usr/include/python3.6/
   /usr/local/lib/python3.6/dist-packages/numpy/core/include/numpy/
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_library(${PROJECT_NAME} SHARED
src/Conversion.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
/usr/lib/x86_64-linux-gnu/libpython3.6m.so
-lglog
)

add_executable(sceneFlow /mnt/SFSS-SLAM/src/SceneFlow/src/SceneFlow.cc)
target_link_libraries(sceneFlow ${PROJECT_NAME})