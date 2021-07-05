cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(VocabularyCreation)

list(APPEND CMAKE_PREFIX_PATH "/home/pyTorchMTP/VocabularyCreation/libtorch/libtorch/")

#find_package(Torch REQUIRED)

# set(TORCH_DIR "/home/pratibha/pyTorchMTP/VocabularyCreation/libtorch/libtorch/share/cmake/Torch")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wall   -O3 -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#this should be above add_executable

find_package(OpenCV 3.2 QUIET)

add_library(${PROJECT_NAME} SHARED
	src/test.cc
)
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/libtorch/include
${PROJECT_SOURCE_DIR}/DBoW2/include/DBoW2
${PROJECT_SOURCE_DIR}/DBoW2/DUtils
${PROJECT_SOURCE_DIR}/DBoW2
${PROJECT_SOURCE_DIR}/DLib/include
${PROJECT_SOURCE_DIR}/DLib/include/DVision
${PROJECT_SOURCE_DIR}/libtorch/include/torch/csrc/api/include/
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/DBoW2/build/libDBoW2.so
${PROJECT_SOURCE_DIR}/libtorch/lib/libtorch.so
${PROJECT_SOURCE_DIR}/libtorch/lib/libtorch_cpu.so
${PROJECT_SOURCE_DIR}/libtorch/lib/libc10.so
${PROJECT_SOURCE_DIR}/DBoW2/build/libDBoW2.so
)


add_executable(create_voc_cdbin create_voc_cdbin.cc)
set_property(TARGET create_voc_cdbin PROPERTY CXX_STANDARD 14)
target_link_libraries(create_voc_cdbin ${PROJECT_NAME})

add_executable(create_voc_gcn create_voc_gcn.cc)
set_property(TARGET create_voc_gcn PROPERTY CXX_STANDARD 14)
target_link_libraries(create_voc_gcn ${PROJECT_NAME})

