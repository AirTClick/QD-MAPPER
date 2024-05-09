project(eecbs)

cmake_minimum_required (VERSION 2.6)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

include_directories("inc")
file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "src/WarehouseSimulation.cpp")
# add_executable(eecbs ${SOURCES})

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
include_directories( ${Boost_INCLUDE_DIRS} )
# target_link_libraries(eecbs ${Boost_LIBRARIES})

# Add pybind11 module
add_subdirectory(pybind11)

# Need to add other source files so that pybind11 can find the implementations
pybind11_add_module(warehouse_sim_EECBS "src/WarehouseSimulation.cpp" ${SOURCES})

target_link_libraries(warehouse_sim_EECBS PRIVATE ${Boost_LIBRARIES})