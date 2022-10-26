cmake_minimum_required(VERSION 3.13)

#defining project name
project(AVG_SLAM)

get_filename_component(AVG_SLAM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/avg_slam DIRECTORY)
message("Project root is:  ${AVG_SLAM_ROOT}")
message("Project name: ${PROJECT_NAME}")
message("CMake binary dir: ${CMAKE_BINARY_DIR}")

# Setting project variables
if (${pangolin_active})
  add_compile_definitions(PANGOLIN_ACTIVE)
endif()

# Subfolders
set(TRACKING_DIR ${AVG_SLAM_ROOT}/tracking)
set(MOTION_PRIOR_DIR ${AVG_SLAM_ROOT}/motionPrior)
set(KEYPOINT_EXTRACTION_DIR ${AVG_SLAM_ROOT}/keypointExtraction)
set(KEYPOINT_MATCHING_DIR ${AVG_SLAM_ROOT}/keypointMatching)
set(POSE_CALCULATION_DIR ${AVG_SLAM_ROOT}/poseCalculation)
set(MAPPOINT_HANDLER_DIR ${AVG_SLAM_ROOT}/mapPointHandler)
set(DATASTRUCTURES_DIR ${AVG_SLAM_ROOT}/dataStructures)
set(SEQUENCER_DIR ${AVG_SLAM_ROOT}/sequencer)
set(UTIL_DIR ${AVG_SLAM_ROOT}/util)
set(TEST_DIR ${AVG_SLAM_ROOT}/test)
if(${pangolin_active})
  set(GUI_DIR ${AVG_SLAM_ROOT}/gui)
endif()

# Subsubfolder
set(MOTION_PRIOR_METHODS_DIR ${AVG_SLAM_ROOT}/motionPrior/motionPriorMethods)
set(KEYPOINT_EXTRACTION_METHODS_DIR ${AVG_SLAM_ROOT}/keypointExtraction/extractionMethods)
set(MATCHING_METHODS_DIR ${AVG_SLAM_ROOT}/keypointMatching/matchingMethods)
set(POSE_CALCULATION_METHODS_DIR ${AVG_SLAM_ROOT}/poseCalculation/poseCalculationMethods)
set(MAP_POINT_REGISTRATION_METHODS_DIR ${AVG_SLAM_ROOT}/mapPointHandler/mapPointRegistrationMethods)
set(TRIANGULATION_METHODS_DIR ${AVG_SLAM_ROOT}/mapPointHandler/triangulationMethods)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


add_compile_options(-std=c++17 -pthread -O0)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_STATIC_LIBRARY_PREFIX_CXX "")
#set(CMAKE_CXX_FLAGS "-Werror -undefined dynamic_lookup")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(DPYTHON_EXECUTABLE /usr/bin/python3)
#set(DPYTHON_LIBRARY /usr/bin/python3)
#set(DPYTHON_INCLUDE_DIR /usr/bin/python3)

set(PROJECT_LIB ${PROJECT_NAME}_lib)
set(AVG_SLAM_MAIN_OBJS
    ${AVG_SLAM_ROOT}/main.cpp
    ${AVG_SLAM_ROOT}/CMakeLists.txt
)

add_library(${PROJECT_NAME}_lib STATIC ${AVG_SLAM_MAIN_OBJS} ${AVG_SLAM_SRC})
add_subdirectory(${TRACKING_DIR})
add_subdirectory(${MOTION_PRIOR_DIR})
add_subdirectory(${KEYPOINT_EXTRACTION_DIR})
add_subdirectory(${KEYPOINT_MATCHING_DIR})
add_subdirectory(${POSE_CALCULATION_DIR})
add_subdirectory(${MAPPOINT_HANDLER_DIR})
add_subdirectory(${DATASTRUCTURES_DIR})
add_subdirectory(${SEQUENCER_DIR})
add_subdirectory(${UTIL_DIR})
add_subdirectory(${TEST_DIR})
if(${pangolin_active})
  add_subdirectory(${GUI_DIR})
endif()

add_subdirectory(${MOTION_PRIOR_METHODS_DIR})
add_subdirectory(${KEYPOINT_EXTRACTION_METHODS_DIR})
add_subdirectory(${MATCHING_METHODS_DIR})
add_subdirectory(${POSE_CALCULATION_METHODS_DIR})
add_subdirectory(${MAP_POINT_REGISTRATION_METHODS_DIR})
add_subdirectory(${TRIANGULATION_METHODS_DIR})


add_executable(${PROJECT_NAME} ${AVG_SLAM_MAIN_OBJS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} TrackingLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} MotionPriorLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} KeypointExtractionLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} KeypointMatchingLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} PoseCalculationLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} MapPointHandlerLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} DataStructuresLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} SequencerLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} UtilLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} TestLib)
if(${pangolin_active})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} GUILib)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} MotionPriorMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} KeypointExtractionMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} MatchingMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} PoseCalculationMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} MapPointRegistrationMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} TriangulationMethodsLib)

target_link_libraries(${PROJECT_NAME} PUBLIC ${CONAN_LIBS})



#find_package(OpenMP REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found")
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()


# Including packages manually if needed
if(${pangolin_active})

  #INCLUDING X11 MANUALLY
  find_package(X11 REQUIRED)

  message(STATUS "X11_FOUND = ${X11_FOUND}")
  message(STATUS "X11_INCLUDE_DIR = ${X11_INCLUDE_DIR}")
  message(STATUS "X11_LIBRARIES = ${X11_LIBRARIES}")

  include_directories(${X11_INCLUDE_DIR})

  link_directories(${X11_LIBRARIES})

  target_link_libraries(${PROJECT_NAME} PUBLIC ${X11_LIBRARIES})
endif()