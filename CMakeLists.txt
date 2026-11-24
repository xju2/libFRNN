cmake_minimum_required(VERSION 3.20)

project(libFRNN VERSION 3.1.1 LANGUAGES CXX CUDA)

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
# AUTO builds the PyTorch bindings when the Python building the package can
# import torch (e.g. pip install --no-build-isolation in an environment with
# torch); ON requires them; OFF skips them.
set(FRNN_BUILD_TORCH "AUTO" CACHE STRING
    "Build the PyTorch CUDA bindings: AUTO (when torch is importable), ON, or OFF")
set_property(CACHE FRNN_BUILD_TORCH PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${FRNN_BUILD_TORCH}" FRNN_BUILD_TORCH_MODE)

if(NOT FRNN_BUILD_TORCH_MODE STREQUAL "AUTO" AND FRNN_BUILD_TORCH AND NOT FRNN_BUILD_PYTHON)
    message(FATAL_ERROR "FRNN_BUILD_TORCH=ON requires FRNN_BUILD_PYTHON=ON")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(frnn src/frnn.cu src/fast_path.cu)
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

# The C++ library, headers and CMake package are installed for C++ users, not
# into Python wheels (scikit-build-core sets SKBUILD), where they would land in
# site-packages.
if(NOT SKBUILD)
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
endif()

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
    add_executable(frnn_pca_agreement benchmarks/pca_agreement.cpp)
    target_link_libraries(frnn_pca_agreement PRIVATE frnn::frnn)
    target_compile_features(frnn_pca_agreement PRIVATE cxx_std_17)
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
    target_compile_definitions(_frnn PRIVATE FRNN_VERSION="${PROJECT_VERSION}")
    install(TARGETS _frnn LIBRARY DESTINATION frnn_cuda)
endif()

# Decide whether to build the PyTorch bindings (see FRNN_BUILD_TORCH above).
set(FRNN_WITH_TORCH OFF)
if(FRNN_BUILD_PYTHON AND (FRNN_BUILD_TORCH_MODE STREQUAL "AUTO" OR FRNN_BUILD_TORCH))
    if(NOT Torch_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                    "import torch; print(torch.utils.cmake_prefix_path)"
            RESULT_VARIABLE FRNN_TORCH_PREFIX_RESULT
            OUTPUT_VARIABLE FRNN_TORCH_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(FRNN_TORCH_PREFIX_RESULT EQUAL 0)
            list(PREPEND CMAKE_PREFIX_PATH "${FRNN_TORCH_PREFIX}")
        endif()
    endif()
    if(FRNN_BUILD_TORCH_MODE STREQUAL "AUTO")
        find_package(Torch CONFIG QUIET)
    else()
        find_package(Torch REQUIRED CONFIG)
    endif()
    # The extension uses PyTorch's CUDA API, so it needs a CUDA build of PyTorch.
    # Only those ship libc10_cuda (CPU-only and ROCm builds do not); checking for
    # it works for pip-installed PyTorch and standalone LibTorch, without a GPU.
    if(Torch_FOUND)
        find_library(FRNN_TORCH_C10_CUDA c10_cuda
                     PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
    endif()
    if(Torch_FOUND AND FRNN_TORCH_C10_CUDA)
        message(STATUS "libFRNN: CUDA PyTorch found, building frnn_cuda.torch")
        set(FRNN_WITH_TORCH ON)
    elseif(NOT FRNN_BUILD_TORCH_MODE STREQUAL "AUTO")
        message(FATAL_ERROR "FRNN_BUILD_TORCH=ON needs a CUDA build of PyTorch, but "
                            "${TORCH_INSTALL_PREFIX} has no c10_cuda library")
    elseif(Torch_FOUND)
        message(STATUS "libFRNN: PyTorch at ${TORCH_INSTALL_PREFIX} is not a CUDA "
                       "build, skipping frnn_cuda.torch")
    else()
        message(STATUS "libFRNN: PyTorch not importable by ${Python_EXECUTABLE}, "
                       "skipping frnn_cuda.torch (build with --no-build-isolation "
                       "in an environment with torch to include it)")
    endif()
endif()

if(FRNN_WITH_TORCH)
    separate_arguments(FRNN_TORCH_CXX_FLAGS NATIVE_COMMAND
                       "${TORCH_CXX_FLAGS}")
    Python_add_library(_frnn_torch MODULE WITH_SOABI python/torch_bindings.cpp)
    target_link_libraries(_frnn_torch PRIVATE frnn::frnn "${TORCH_LIBRARIES}")
    target_compile_features(_frnn_torch PRIVATE cxx_std_17)
    target_compile_options(_frnn_torch PRIVATE ${FRNN_TORCH_CXX_FLAGS})
    install(TARGETS _frnn_torch LIBRARY DESTINATION frnn_cuda)
endif()
