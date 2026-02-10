cmake_minimum_required(VERSION 4.1)
project(fluxions)

set(CMAKE_CXX_STANDARD 20)

add_executable(fluxions main.cpp
        memory/memory.h)
