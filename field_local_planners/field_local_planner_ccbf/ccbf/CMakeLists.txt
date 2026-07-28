cmake_minimum_required(VERSION 3.10)
project(ccbf VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the build type." FORCE)
endif()

# Catkin passes CATKIN_DEVEL_PREFIX when it configures a package. Checking this
# instead of searching for Catkin keeps plain CMake builds standalone even when
# the user's shell has sourced a ROS environment.
set(CCBF_CATKIN_BUILD OFF)
if(DEFINED CATKIN_DEVEL_PREFIX)
  set(CCBF_CATKIN_BUILD ON)
  find_package(catkin REQUIRED)
endif()

find_package(Eigen3 REQUIRED)
find_package(proxsuite REQUIRED)

if(CCBF_CATKIN_BUILD)
  # Catkin's non-Catkin dependency export expects this conventional variable,
  # while modern Eigen exports only the Eigen3::Eigen target.
  if(NOT Eigen3_INCLUDE_DIRS)
    get_target_property(
      Eigen3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  endif()

  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    DEPENDS Eigen3 proxsuite
  )
else()
  include(CTest)
  include(GNUInstallDirs)
endif()

add_library(${PROJECT_NAME} SHARED
  src/dynamics.cpp
  src/barrier.cpp
  src/qp_builder.cpp
  src/proxqp_solver.cpp
  src/cbf_controller.cpp
)
add_library(ccbf::ccbf ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    proxsuite::proxsuite
)

if(CCBF_CATKIN_BUILD)
  target_include_directories(${PROJECT_NAME} PUBLIC ${catkin_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES})
endif()

add_executable(${PROJECT_NAME}_demo examples/ccbf_demo.cpp)
target_link_libraries(${PROJECT_NAME}_demo PRIVATE ccbf::ccbf)

add_executable(${PROJECT_NAME}_simulation examples/ccbf_simulation.cpp)
target_link_libraries(${PROJECT_NAME}_simulation PRIVATE ccbf::ccbf)

if(CCBF_CATKIN_BUILD)
  if(CATKIN_ENABLE_TESTING)
    catkin_add_gtest(${PROJECT_NAME}_test
      test/main.cpp
      test/ccbf_test.cpp
    )
    if(TARGET ${PROJECT_NAME}_test)
      # catkin_add_gtest uses the plain target_link_libraries signature.
      target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})
    endif()

    add_test(
      NAME ${PROJECT_NAME}_demo_smoke
      COMMAND $<TARGET_FILE:${PROJECT_NAME}_demo>
    )
    add_test(
      NAME ${PROJECT_NAME}_simulation_smoke
      COMMAND $<TARGET_FILE:${PROJECT_NAME}_simulation>
              ${CMAKE_CURRENT_BINARY_DIR}/ccbf_simulation.csv
    )
  endif()

  install(
    TARGETS
      ${PROJECT_NAME}
      ${PROJECT_NAME}_demo
      ${PROJECT_NAME}_simulation
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(
    DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp"
  )

  install(
    PROGRAMS examples/plot_simulation.py
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/examples
  )
else()
  if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_executable(${PROJECT_NAME}_test
      test/main.cpp
      test/ccbf_test.cpp
    )
    target_link_libraries(${PROJECT_NAME}_test
      PRIVATE
        ccbf::ccbf
        GTest::gtest
    )

    add_test(NAME ${PROJECT_NAME}_unit COMMAND ${PROJECT_NAME}_test)
    add_test(NAME ${PROJECT_NAME}_demo_smoke COMMAND ${PROJECT_NAME}_demo)
    add_test(
      NAME ${PROJECT_NAME}_simulation_smoke
      COMMAND ${PROJECT_NAME}_simulation
              ${CMAKE_CURRENT_BINARY_DIR}/ccbf_simulation.csv
    )
  endif()

  install(
    TARGETS
      ${PROJECT_NAME}
      ${PROJECT_NAME}_demo
      ${PROJECT_NAME}_simulation
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
  )

  install(
    PROGRAMS examples/plot_simulation.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/ccbf/examples
  )

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    cmake/ccbfConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ccbfConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccbf
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ccbfConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ccbfTargets.cmake
    NAMESPACE ccbf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccbf
  )
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/ccbfConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/ccbfConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ccbf
  )
endif()
