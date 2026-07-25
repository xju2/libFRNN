find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# CUDA toolkit wheels do not put nvcc on PATH. If no toolkit was selected
# explicitly, ask cuda-pathfinder (installed with CUDA-enabled PyTorch wheels)
# for the matching compiler before Torch enables the CUDA language.
if(NOT CMAKE_CUDA_COMPILER
   AND NOT DEFINED ENV{CUDACXX}
   AND NOT CUDA_TOOLKIT_ROOT_DIR
   AND NOT CUDAToolkit_ROOT)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "from cuda.pathfinder import find_nvidia_binary_utility; print(find_nvidia_binary_utility(\"nvcc\"))"
        RESULT_VARIABLE FRNN_NVCC_RESULT
        OUTPUT_VARIABLE FRNN_NVCC
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(FRNN_NVCC_RESULT EQUAL 0 AND EXISTS "${FRNN_NVCC}")
        get_filename_component(FRNN_CUDA_BIN_DIR "${FRNN_NVCC}" DIRECTORY)
        get_filename_component(FRNN_CUDA_ROOT "${FRNN_CUDA_BIN_DIR}" DIRECTORY)
        set(CMAKE_CUDA_COMPILER "${FRNN_NVCC}" CACHE FILEPATH
            "CUDA compiler" FORCE)
        set(CUDA_TOOLKIT_ROOT_DIR "${FRNN_CUDA_ROOT}" CACHE PATH
            "CUDA toolkit root" FORCE)
        set(CUDAToolkit_ROOT "${FRNN_CUDA_ROOT}" CACHE PATH
            "CUDA toolkit root" FORCE)
        message(STATUS "Using CUDA compiler from Python environment: ${FRNN_NVCC}")
    endif()
endif()

find_package(Torch REQUIRED)

set(FRNN_CXX_FLAGS "${TORCH_CXX_FLAGS}")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND FRNN_CXX_FLAGS -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)

set(FRNN_CUDA_FLAGS "")
list(APPEND FRNN_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND FRNN_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
list(APPEND FRNN_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND FRNN_CUDA_FLAGS -Xfatbin=-compress-all)
if (CMAKE_CUDA_LINEINFO)
    list(APPEND CUGRAPH_CUDA_FLAGS -lineinfo)
endif(CMAKE_CUDA_LINEINFO)
# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND FRNN_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)


file(GLOB SrcFiles *.cu)
add_library(frnn SHARED ${SrcFiles})

target_include_directories(frnn
    PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(frnn PUBLIC torch Python3::Python)


set_property(TARGET frnn PROPERTY CXX_STANDARD 17)
set_property(TARGET frnn PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET frnn PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET frnn PROPERTY CUDA_STANDARD 17)

target_compile_options(frnn
    PUBLIC  "$<$<COMPILE_LANGUAGE:CXX>:${FRNN_CXX_FLAGS}>"
            "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${FRNN_CUDA_FLAGS}>>"
)

install(TARGETS frnn 
    DESTINATION lib
    EXPORT frnnTargets)

file(GLOB HeaderFiles
    include/frnn/*.h
    include/frnn/*.hpp
    include/frnn/*.cuh
)
install(FILES ${HeaderFiles} DESTINATION include/frnn)
