cmake_minimum_required (VERSION 2.8) 
project(haar_training)

# Include the pkg config module
include(FindPkgConfig)
# Find the OpenCV Module
pkg_check_modules(OpenCV REQUIRED "opencv")

if(${OpenCV_FOUND} )
  
  set(${CMAKE_BUILD_TYPE} Release)
  
  # Output the binaries in the bin subfolder.
  set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

  add_executable(${PROJECT_NAME}_rectangle.out src/rectangle.cpp) 
  target_link_libraries(${PROJECT_NAME}_rectangle.out ${OpenCV_LDFLAGS})

  add_executable(${PROJECT_NAME}_capture.out src/capture.cpp) 
  target_link_libraries(${PROJECT_NAME}_capture.out ${OpenCV_LDFLAGS})
 
  # Include the project's headers and the necessary opencv headers.
  include_directories(include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})
  # Add the necessary linker flags to link our project's source files
  # with the OpenCV libs.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENCV_CFLAGS}")

  # Create the executable that merges the positive and the negative examples
  # in a single file.
  add_executable(${PROJECT_NAME}_mergevec.out
    src/mergevec.cpp
    src/cvboost.cpp
    src/cvhaarclassifier.cpp
    src/cvsamples.cpp
    src/cvcommon.cpp
    src/cvhaartraining.cpp
  )
  
  # Link the executable to the necessary libraries.
  target_link_libraries(${PROJECT_NAME}_mergevec.out 
    ${OpenCV_LIBRARIES}
    ${OpenCV_LDFLAGS})
  
elseif(NOT ${OpenCV_FOUND})
  message(FATAL_ERROR "Could not find OpenCV package!")
elseif(NOT ${Boost_FOUND})
  message(FATAL_ERROR "Could not find Boost package!")

endif()


