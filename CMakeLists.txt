cmake_minimum_required (VERSION 3.4)

project(lifelong)

# Let cmake build for python 3.9
find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)


IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
    #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
    #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")


include_directories("EECBS/inc")
# include_directories("RHCR/inc")

# Get the source files except for the main API .cpp file
file(GLOB SOURCES "EECBS/src/*.cpp")
# file(GLOB SOURCES "RHCR/src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "EECBS/src/WarehouseSimulation.cpp")
# list(FILTER SOURCES EXCLUDE REGEX "RHCR/src/WarehouseSimulation.cpp")

# Find Boost
find_package( Boost 1.71.0 REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )

# Add pybind11 module
add_subdirectory(pybind11)

# Need to add other source files so that pybind11 can find the implementations
pybind11_add_module(warehouse_sim_EECBS "EECBS/src/WarehouseSimulation.cpp" ${SOURCES})
# pybind11_add_module(warehouse_sim "RHCR/src/WarehouseSimulation.cpp" ${SOURCES})
target_link_libraries(warehouse_sim_EECBS PRIVATE ${Boost_LIBRARIES})

# # Add lifelone executable
# add_executable(lifelong ${SOURCES} )
# target_link_libraries(lifelong ${Boost_LIBRARIES})


# ============================================================================ #
# project(lifelong)

# cmake_minimum_required (VERSION 2.6)
# set(CMAKE_CXX_STANDARD 11)

# IF(NOT CMAKE_BUILD_TYPE)
#     #SET(CMAKE_BUILD_TYPE "DEBUG")
#     SET(CMAKE_BUILD_TYPE "RELEASE")
#     #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
#     #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
# ENDIF()

# include_directories("RHCR/inc")

# # Get the source files except for the main API .cpp file
# file(GLOB SOURCES "RHCR/src/*.cpp")
# list(FILTER SOURCES EXCLUDE REGEX "RHCR/src/WarehouseSimulation.cpp")

# # Find Boost
# find_package( Boost 1.49.0 REQUIRED COMPONENTS program_options system filesystem)
# include_directories( ${Boost_INCLUDE_DIRS} )


# add_executable(lifelong ${SOURCES} )
# target_link_libraries(lifelong ${Boost_LIBRARIES})
