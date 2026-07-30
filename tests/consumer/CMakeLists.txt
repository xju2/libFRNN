cmake_minimum_required(VERSION 3.20)
project(frnn_consumer LANGUAGES CXX)

find_package(frnn CONFIG REQUIRED)

add_executable(frnn_consumer main.cpp)
target_link_libraries(frnn_consumer PRIVATE frnn::frnn)
target_compile_features(frnn_consumer PRIVATE cxx_std_17)
