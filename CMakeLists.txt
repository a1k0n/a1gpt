# Set minimum version of CMake. Version 3.8 introduces CUDA as a first-class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(my_project LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_SHALLOW    TRUE
  GIT_PROGRESS   TRUE
  GIT_TAG        v3.9.1
)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")

# Add executable target with source files
add_executable(thingy main.cpp kernel.cu)
target_include_directories(thingy PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(thingy PRIVATE nlohmann_json::nlohmann_json)

# Specify the C++ standard
set_target_properties(thingy PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED TRUE)

