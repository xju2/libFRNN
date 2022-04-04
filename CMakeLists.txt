cmake_minimum_required (VERSION 3.20)
project (libFRNN VERSION 1.0)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

add_subdirectory(src)

install(EXPORT frnnTargets
    FILE frnnTargets.cmake
    DESTINATION lib/cmake/frnn
)

include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/frnnConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/example"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/frnnConfigVersion.cmake"
  VERSION "${libFRNN_VERSION_MAJOR}.${libFRNN_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)
# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/frnnConfig.cmake
  DESTINATION lib/cmake/frnn
)

export(EXPORT frnnTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/frnnTargets.cmake"
)

include(InstallRequiredSystemLibraries)
# set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${libFRNN_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${libFRNN_VERSION_MINOR}")
include(CPack)