cmake_minimum_required(VERSION 3.13)

#defining project name
project(AVG_SLAM)

get_filename_component(AVG_SLAM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/avg_slam DIRECTORY)
message("Project root is:  ${AVG_SLAM_ROOT}")
message("Project name: ${PROJECT_NAME}")
message("CMake binary dir: ${CMAKE_BINARY_DIR}")

# Setting project variables
add_compile_definitions(PANGOLIN_ACTIVE ${pangolin_active})
# Subfolders
set(TRACKING_DIR ${AVG_SLAM_ROOT}/tracking)
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
set(MATCHING_METHODS_DIR ${AVG_SLAM_ROOT}/keypointMatching/matchingMethods)
set(POSE_CALCULATION_METHODS_DIR ${AVG_SLAM_ROOT}/poseCalculation/poseCalculationMethods)
set(MAP_POINT_REGISTRATION_METHODS_DIR ${AVG_SLAM_ROOT}/mapPointHandler/mapPointRegistrationMethods)
set(TRIANGULATION_METHODS_DIR ${AVG_SLAM_ROOT}/mapPointHandler/triangulationMethods)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_STATIC_LIBRARY_PREFIX_CXX "")
#set (CMAKE_CXX_FLAGS "-Werror -undefined dynamic_lookup")
#set (CMAKE_POSITION_INDEPENDENT_CODE ON)
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

add_subdirectory(${MATCHING_METHODS_DIR})
add_subdirectory(${POSE_CALCULATION_METHODS_DIR})
add_subdirectory(${MAP_POINT_REGISTRATION_METHODS_DIR})
add_subdirectory(${TRIANGULATION_METHODS_DIR})


add_executable(${PROJECT_NAME} ${AVG_SLAM_MAIN_OBJS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} TrackingLib)
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

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} MatchingMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} PoseCalculationMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} MapPointRegistrationMethodsLib)
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIB} TriangulationMethodsLib)

target_link_libraries(${PROJECT_NAME} PUBLIC ${CONAN_LIBS})


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




  if(${os_wsl})
    #INCLUDING OPENNI MANUALLY

    find_package(PkgConfig QUIET)
    pkg_check_modules(PC_OPENNI QUIET libopenni)

    set(OPENNI_DEFINITIONS ${PC_OPENNI_CFLAGS_OTHER})

    set(OPENNI_SUFFIX)
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(OPENNI_SUFFIX 64)
    endif()

    # Add a hint so that it can find it without the pkg-config
    find_path(OPENNI_INCLUDE_DIR XnStatus.h
              HINTS ${PC_OPENNI_INCLUDEDIR}
                    ${PC_OPENNI_INCLUDE_DIRS}
                    /usr/include/openni
                    /usr/include/ni
                    /opt/local/include/ni
                    "${OPENNI_ROOT}"
                    "$ENV{OPENNI_ROOT}"
              PATHS "$ENV{OPEN_NI_INSTALL_PATH${OPENNI_SUFFIX}}/Include"
              PATH_SUFFIXES openni include Include)

    # Add a hint so that it can find it without the pkg-config
    find_library(OPENNI_LIBRARY
                NAMES OpenNI${OPENNI_SUFFIX}
                HINTS ${PC_OPENNI_LIBDIR}
                      ${PC_OPENNI_LIBRARY_DIRS}
                      /usr/lib
                      "${OPENNI_ROOT}"
                      "$ENV{OPENNI_ROOT}"
                PATHS "$ENV{OPEN_NI_LIB${OPENNI_SUFFIX}}"
                PATH_SUFFIXES lib Lib Lib64)

    if(OPENNI_INCLUDE_DIR AND OPENNI_LIBRARY)

      # Include directories
      set(OPENNI_INCLUDE_DIRS ${OPENNI_INCLUDE_DIR})
      unset(OPENNI_INCLUDE_DIR)
      mark_as_advanced(OPENNI_INCLUDE_DIRS)

      # Libraries
      if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        find_package(libusb REQUIRED)
        set(OPENNI_LIBRARIES ${OPENNI_LIBRARY} libusb::libusb)
      else()
        set(OPENNI_LIBRARIES ${OPENNI_LIBRARY})
      endif()
      unset(OPENNI_LIBRARY)
      mark_as_advanced(OPENNI_LIBRARIES)

    endif()

    if(EXISTS "${OPENNI_INCLUDE_DIR}/XnVersion.h")
      file(STRINGS "${OPENNI_INCLUDE_DIR}/XnVersion.h" _contents REGEX "^#define[ \t]+XN_[A-Z]+_VERSION[ \t]+[0-9]+")
      if(_contents)
        string(REGEX REPLACE ".*#define[ \t]+XN_MAJOR_VERSION[ \t]+([0-9]+).*" "\\1" OPENNI_VERSION_MAJOR "${_contents}")
        string(REGEX REPLACE ".*#define[ \t]+XN_MINOR_VERSION[ \t]+([0-9]+).*" "\\1" OPENNI_VERSION_MINOR "${_contents}")
        string(REGEX REPLACE ".*#define[ \t]+XN_MAINTENANCE_VERSION[ \t]+([0-9]+).*" "\\1" OPENNI_VERSION_PATCH "${_contents}")
        string(REGEX REPLACE ".*#define[ \t]+XN_BUILD_VERSION[ \t]+([0-9]+).*" "\\1" OPENNI_VERSION_BUILD "${_contents}")
        set(OPENNI_VERSION "${OPENNI_VERSION_MAJOR}.${OPENNI_VERSION_MINOR}.${OPENNI_VERSION_PATCH}.${OPENNI_VERSION_BUILD}")
      endif()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(OpenNI
      FOUND_VAR OPENNI_FOUND
      REQUIRED_VARS OPENNI_LIBRARIES OPENNI_INCLUDE_DIRS
      VERSION_VAR OPENNI_VERSION
    )

    if(OPENNI_FOUND)
      message(STATUS "OpenNI found (version: ${OPENNI_VERSION}, include: ${OPENNI_INCLUDE_DIRS}, lib: ${OPENNI_LIBRARIES})")
    endif()

    message(STATUS "OPENNI_FOUND = ${OPENNI_FOUND}")
    message(STATUS "OPENNI_INCLUDE_DIR = ${OPENNI_INCLUDE_DIR}")
    message(STATUS "OPENNI_LIBRARIES = ${OPENNI_LIBRARIES}")

    include_directories(${OPENNI_INCLUDE_DIR})

    link_directories(${OPENNI_LIBRARIES})

    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENNI_LIBRARIES})



    #INCLUDING OPENNI2
    find_package(PkgConfig QUIET)
    pkg_check_modules(PC_OPENNI2 QUIET libopenni2)

    set(OPENNI2_DEFINITIONS ${PC_OPENNI_CFLAGS_OTHER})

    set(OPENNI2_SUFFIX)
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(OPENNI2_SUFFIX 64)
    endif()

    find_path(OPENNI2_INCLUDE_DIR OpenNI.h
              PATHS "$ENV{OPENNI2_INCLUDE${OPENNI2_SUFFIX}}"  # Win64 needs '64' suffix
                    "/usr/include/openni2"                    # common path for deb packages
              PATH_SUFFIXES include/openni2
    )

    find_library(OPENNI2_LIBRARY
                NAMES OpenNI2      # No suffix needed on Win64
                      libOpenNI2   # Linux
                PATHS "$ENV{OPENNI2_LIB${OPENNI2_SUFFIX}}"   # Windows default path, Win64 needs '64' suffix
                      "$ENV{OPENNI2_REDIST}"                 # Linux install does not use a separate 'lib' directory
    )

    if(OPENNI2_INCLUDE_DIR AND OPENNI2_LIBRARY)

      # Include directories
      set(OPENNI2_INCLUDE_DIRS ${OPENNI2_INCLUDE_DIR})
      unset(OPENNI2_INCLUDE_DIR)
      mark_as_advanced(OPENNI2_INCLUDE_DIRS)

      # Libraries
      if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        find_package(libusb REQUIRED)
        set(OPENNI2_LIBRARIES ${OPENNI2_LIBRARY} libusb::libusb)
      else()
        set(OPENNI2_LIBRARIES ${OPENNI2_LIBRARY})
      endif()
      unset(OPENNI2_LIBRARY)
      mark_as_advanced(OPENNI2_LIBRARIES)

      set(OPENNI2_REDIST_DIR $ENV{OPENNI2_REDIST${OPENNI2_SUFFIX}})
      mark_as_advanced(OPENNI2_REDIST_DIR)

    endif()

    if(EXISTS "${OPENNI2_INCLUDE_DIR}/OniVersion.h")
      file(STRINGS "${OPENNI2_INCLUDE_DIR}/OniVersion.h" _contents REGEX "^#define[ \t]+ONI_VERSION_[A-Z]+[ \t]+[0-9]+")
      if(_contents)
        string(REGEX REPLACE ".*#define[ \t]+ONI_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" OPENNI2_VERSION_MAJOR "${_contents}")
        string(REGEX REPLACE ".*#define[ \t]+ONI_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" OPENNI2_VERSION_MINOR "${_contents}")
        string(REGEX REPLACE ".*#define[ \t]+ONI_VERSION_MAINTENANCE[ \t]+([0-9]+).*" "\\1" OPENNI2_VERSION_PATCH "${_contents}")
        string(REGEX REPLACE ".*#define[ \t]+ONI_VERSION_BUILD[ \t]+([0-9]+).*" "\\1" OPENNI2_VERSION_BUILD "${_contents}")
        set(OPENNI2_VERSION "${OPENNI2_VERSION_MAJOR}.${OPENNI2_VERSION_MINOR}.${OPENNI2_VERSION_PATCH}.${OPENNI2_VERSION_BUILD}")
      endif()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(OpenNI2
      FOUND_VAR OPENNI2_FOUND
      REQUIRED_VARS OPENNI2_LIBRARIES OPENNI2_INCLUDE_DIRS
      VERSION_VAR OPENNI2_VERSION
    )

    if(OPENNI2_FOUND)
      message(STATUS "OpenNI2 found (version: ${OPENNI2_VERSION}, include: ${OPENNI2_INCLUDE_DIRS}, lib: ${OPENNI2_LIBRARIES})")
    endif()

    message(STATUS "OPENNI2_FOUND = ${OPENNI2_FOUND}")
    message(STATUS "OPENNI2_INCLUDE_DIR = ${OPENNI2_INCLUDE_DIR}")
    message(STATUS "OPENNI2_LIBRARIES = ${OPENNI2_LIBRARIES}")

    include_directories(${OPENNI2_INCLUDE_DIR})

    link_directories(${OPENNI2_LIBRARIES})

    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENNI2_LIBRARIES})
  endif()
endif()


#add_executable(helloExample helloExample.cpp)
#target_link_libraries(helloExample ${CONAN_LIBS})