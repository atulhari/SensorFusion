cmake_minimum_required(VERSION 3.0.2)
project(sfip)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  cmake_modules
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Add message files
add_message_files(
  FILES
  Estimate.msg
  Knot.msg
  Spline.msg
)

# Generate messages with dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

find_package(Eigen3 REQUIRED)

# Find SuiteSparse if available, but don't fail if not found
find_package(SuiteSparse QUIET)
if(SUITESPARSE_FOUND)
  message(STATUS "SuiteSparse found. Core optimization functionality will be enabled.")
  add_definitions(-DWITH_SUITESPARSE)
else()
  message(WARNING "SuiteSparse NOT found. Core optimization will be disabled, but tests and visualization can still run.")
endif()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs message_runtime tf2 tf2_ros tf2_geometry_msgs
  DEPENDS EIGEN3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${BOOST_INCLUDE_DIR}
)

# Add SuiteSparse include directories if found
if(SUITESPARSE_FOUND)
  include_directories(${SUITESPARSE_INCLUDE_DIRS})
endif()

# Add the interface executable
add_executable(EstimationInterface_new src/EstimationInterface_new.cpp)
target_link_libraries(EstimationInterface_new ${catkin_LIBRARIES})

add_dependencies(EstimationInterface_new ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Add the figure-8 trajectory test
if(CATKIN_ENABLE_TESTING)
  find_package(GTest REQUIRED)
  catkin_add_gtest(figure8_trajectory_test test/figure8_trajectory_test.cpp)
  target_link_libraries(figure8_trajectory_test ${catkin_LIBRARIES} ${GTEST_LIBRARIES})
  
  # Add the integration test
  catkin_add_gtest(integration_test test/integration_test.cpp)
  target_link_libraries(integration_test ${catkin_LIBRARIES} ${GTEST_LIBRARIES})
  
  # Add the comprehensive integration test
  catkin_add_gtest(comprehensive_integration_test test/comprehensive_integration_test.cpp)
  target_link_libraries(comprehensive_integration_test ${catkin_LIBRARIES} ${GTEST_LIBRARIES})
endif()
