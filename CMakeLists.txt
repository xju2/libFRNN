cmake_minimum_required(VERSION 3.20)

project(libFRNN VERSION 1.0.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE FRNN_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(NOT FRNN_GIT_COMMIT)
    set(FRNN_GIT_COMMIT "unknown")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries instead of static libraries" ON)
option(FRNN_BUILD_TESTS "Build FRNN tests" ON)
option(FRNN_BUILD_BENCHMARKS "Build FRNN benchmarks" OFF)
option(FRNN_BUILD_PYTHON "Build Python bindings" OFF)

find_package(CUDAToolkit REQUIRED)

add_library(frnn src/frnn.cu)
add_library(frnn::frnn ALIAS frnn)
set_target_properties(frnn PROPERTIES
    EXPORT_NAME frnn
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_include_directories(frnn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(frnn PUBLIC CUDA::cudart)
target_compile_options(frnn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

install(TARGETS frnn
    EXPORT frnnTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY src/include/frnn
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)
install(EXPORT frnnTargets
    FILE frnnTargets.cmake
    NAMESPACE frnn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/frnn
)

configure_package_config_file(
    Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/frnnConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/frnn
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/frnnConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/frnnConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/frnnConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/frnn
)

if(FRNN_BUILD_TESTS)
    enable_testing()
    add_executable(frnn_core_tests tests/test_frnn.cpp)
    target_link_libraries(frnn_core_tests PRIVATE frnn::frnn)
    target_compile_features(frnn_core_tests PRIVATE cxx_std_17)
    add_test(NAME frnn_core_tests COMMAND frnn_core_tests)
endif()

if(FRNN_BUILD_BENCHMARKS)
    target_compile_definitions(frnn PRIVATE FRNN_ENABLE_STAGE_PROFILING=1)
    add_executable(frnn_benchmark benchmarks/frnn_benchmark.cu)
    target_link_libraries(frnn_benchmark PRIVATE frnn::frnn)
    target_compile_features(frnn_benchmark PRIVATE cxx_std_17)
    target_compile_definitions(frnn_benchmark PRIVATE
        FRNN_GIT_COMMIT="${FRNN_GIT_COMMIT}"
        FRNN_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        FRNN_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        FRNN_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}"
        FRNN_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
    )
    set_target_properties(frnn_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(FRNN_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    # NO_EXTRAS avoids pybind11's automatic LTO flags. Those flags can be
    # incompatible when nvcc and the selected host C++ compiler use different
    # GCC LTO bytecode versions.
    pybind11_add_module(_frnn MODULE NO_EXTRAS python/bindings.cpp)
    target_link_libraries(_frnn PRIVATE frnn::frnn)
    target_compile_features(_frnn PRIVATE cxx_std_17)
    install(TARGETS _frnn LIBRARY DESTINATION frnn_cuda)
endif()
