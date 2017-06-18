cmake_minimum_required(VERSION 2.8)
project(simulator)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "~/gpu/development/simulator/bin")

include_directories(include)

set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FireSim.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/FireSim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel_BD.cu
#    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel_BD.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel_IMT.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel_MT.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel_common.cu
#    ${CMAKE_CURRENT_SOURCE_DIR}/include/vector.h
#    ${CMAKE_CURRENT_SOURCE_DIR}/include/MT.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/IMT.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/IMT.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/BD.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/BD.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MT.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/include/MT.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/SEQ.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/SEQ.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/propagation.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/propagation.cu
)
#add_library(include firesim.cpp)

find_package(CUDA QUIET REQUIRED)
## nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -std=c++11)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g arch=compute_20,code=sm_20 -std=c++11)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDALIBS ${CUDALIBS} ${CUDA_LIBRARIES})

#cuda_compile(PROPO src/simulation.cu)
#cuda_compile(BDO include/BD.cpp)

find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIR})
set(GDALLIBS ${GDALLIBS} ${GDAL_LIBRARY})

cuda_add_executable(simulator ${SOURCE_FILES})

#add_library(vFireLib SHARED ${SOURCE_FILES})

#target_link_libraries(mylib ${CUDA_LIBRARIES})
target_link_libraries(simulator ${CUDALIBS} ${GDALLIBS})
#target_link_libraries(simulator ${GDALLIBS})
