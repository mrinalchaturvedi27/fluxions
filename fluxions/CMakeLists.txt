cmake_minimum_required(VERSION 4.1)
project(.)

set(CMAKE_CXX_STANDARD 20)

add_library(fluxions_memory
        memory/mem.cc
        memory/mem_pool.cc
)

target_include_directories(fluxions_memory PUBLIC memory)

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
