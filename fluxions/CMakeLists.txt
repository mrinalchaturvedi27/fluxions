cmake_minimum_required(VERSION 4.1)
project(fluxions LANGUAGES CXX)

include(FetchContent)

find_package(Eigen3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
        FetchContent_Declare(
                eigen
                URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
        )
        FetchContent_MakeAvailable(eigen)
endif()

add_library(fluxions_core
        core/tensor.cc
)

target_include_directories(fluxions_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fluxions_core PUBLIC
        Eigen3::Eigen
)

add_library(fluxions_memory
        memory/mem.cc
        memory/mem_pool.cc
)

target_include_directories(fluxions_memory PUBLIC memory)

target_link_libraries(fluxions_memory PUBLIC
        fluxions_core
)

find_package(GTest REQUIRED)

add_executable(test_memory
        tests/test_memory.cpp
)

target_link_libraries(test_memory
        fluxions_memory
        GTest::gtest
        GTest::gtest_main
        pthread
)

set_target_properties(test_memory PROPERTIES
        BUILD_RPATH "/opt/anaconda3/lib"
)


include(GoogleTest)
gtest_discover_tests(test_memory)
