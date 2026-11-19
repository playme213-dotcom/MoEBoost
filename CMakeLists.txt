cmake_minimum_required(VERSION 3.18)
project(moe_pager_cpp LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(moe_pager_cpp cpp_core/moe_pager.cpp)
target_compile_features(moe_pager_cpp PRIVATE cxx_std_17)
target_compile_options(moe_pager_cpp PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(moe_pager_cpp PRIVATE -O3 -march=native)
find_package(Threads REQUIRED)
target_link_libraries(moe_pager_cpp PRIVATE Threads::Threads)
