cmake_minimum_required(VERSION 3.15...3.29)
project(nbody_cpp)
include(FetchContent)



set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(Eigen QUIET)
if(NOT Eigen_FOUND)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        )

    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_MPL2_ONLY ON)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    set(EIGEN_BUILD_DOC OFF)
    FetchContent_MakeAvailable(Eigen)
endif()


pybind11_add_module(nbody_cpp src/nbody.cpp src/nbody_bind.cpp)
target_link_libraries(nbody_cpp PRIVATE Eigen3::Eigen)
install(TARGETS nbody_cpp DESTINATION .)