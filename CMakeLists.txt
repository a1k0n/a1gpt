# Set minimum version of CMake. Version 3.8 introduces CUDA as a first-class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(a1gpt LANGUAGES CXX) # CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

if (APPLE)
    find_library(ACCELERATE Accelerate)
endif()

# Add executable target with source files
add_executable(gpt2 main.cpp
  model.h model.cpp model_load_gpt2.cpp
  blas.h tensor.h
  bpe.h bpe.cpp)
  # kernel.cu)

  # target_include_directories(gpt2 PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpt2 PRIVATE ${ACCELERATE})


# Specify the C++ standard
set_target_properties(gpt2 PROPERTIES
  #                      CUDA_SEPARABLE_COMPILATION ON
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED TRUE)


# add bpe_test
add_executable(bpe_test bpe_test.cpp bpe.h bpe.cpp)
set_target_properties(bpe_test PROPERTIES
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED TRUE)

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                        ${CMAKE_SOURCE_DIR}/model
                        ${CMAKE_BINARY_DIR}/model)

