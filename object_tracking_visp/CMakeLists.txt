cmake_minimum_required(VERSION 3.16)

project(object_tracking_visp LANGUAGES CXX)

option(OBJECT_TRACKING_ENABLE_CUDA "Use CUDA for OpenCV DNN inference" OFF)

find_package(OpenCV REQUIRED COMPONENTS
  core
  imgproc
  highgui
  videoio
  calib3d
  dnn
)
find_package(VISP REQUIRED)

add_executable(object_tracking object_tracking.cpp)
target_compile_features(object_tracking PRIVATE cxx_std_17)
set(VISP_INSTALL_INCLUDE_DIR "${VISP_CONFIG_PATH}/../../../include")
if(EXISTS "${VISP_INSTALL_INCLUDE_DIR}/visp3/core/vpConfig.h")
  list(APPEND VISP_INCLUDE_DIRS "${VISP_INSTALL_INCLUDE_DIR}")
endif()
target_include_directories(object_tracking PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
)
target_link_libraries(object_tracking PRIVATE
  ${OpenCV_LIBS}
  ${VISP_LIBRARIES}
)

if(OBJECT_TRACKING_ENABLE_CUDA)
  target_compile_definitions(object_tracking PRIVATE OBJECT_TRACKING_ENABLE_CUDA)
endif()

if(WIN32)
  foreach(visp_module IN LISTS VISP_LIBS)
    if(TARGET ${visp_module})
      add_custom_command(TARGET object_tracking POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:${visp_module}>"
          "$<TARGET_FILE_DIR:object_tracking>"
        VERBATIM
      )
    endif()
  endforeach()

  file(GLOB OpenCV_RUNTIME_DLLS "${OpenCV_DIR}/../bin/opencv_*.dll")
  foreach(opencv_dll IN LISTS OpenCV_RUNTIME_DLLS)
    add_custom_command(TARGET object_tracking POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${opencv_dll}"
        "$<TARGET_FILE_DIR:object_tracking>"
      VERBATIM
    )
  endforeach()
endif()
