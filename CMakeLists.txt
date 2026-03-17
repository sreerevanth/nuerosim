cmake_minimum_required(VERSION 3.20)
project(neurosim VERSION 1.0.0 LANGUAGES CXX CUDA)

# ─────────────────────────────────────────────────────────────
#  Build options
# ─────────────────────────────────────────────────────────────
option(ENABLE_CUDA   "Build CUDA acceleration kernels"  ON)
option(ENABLE_MPI    "Build MPI distributed support"    ON)
option(ENABLE_OPENMP "Build OpenMP shared memory par."  ON)
option(BUILD_TESTS   "Build unit tests"                 ON)
option(BUILD_PYTHON  "Build Python bindings (pybind11)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────────────────────────────────────────
#  Compiler flags
# ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90")  # V100, A100, RTX30, H100
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --ptxas-options=-v")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    add_compile_definitions(NEUROSIM_CUDA_ENABLED)
endif()

# ─────────────────────────────────────────────────────────────
#  Dependencies
# ─────────────────────────────────────────────────────────────
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Eigen3 3.4 REQUIRED)

if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(NEUROSIM_MPI_ENABLED)
endif()

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

# ─────────────────────────────────────────────────────────────
#  Core C++ library
# ─────────────────────────────────────────────────────────────
set(CORE_SOURCES
    cpp/src/simulation_engine.cpp
    cpp/src/hodgkin_huxley.cpp
    cpp/src/synapse_model.cpp
    cpp/src/neurotransmitter.cpp
    cpp/src/network_solver.cpp
)

if(ENABLE_CUDA)
    list(APPEND CORE_SOURCES
        cpp/kernels/hh_cuda.cu
        cpp/kernels/synapse_cuda.cu
        cpp/kernels/diffusion_cuda.cu
    )
endif()

if(ENABLE_MPI)
    list(APPEND CORE_SOURCES
        cpp/mpi/distributed_sim.cpp
        cpp/mpi/partition_manager.cpp
    )
endif()

add_library(neurosim_core SHARED ${CORE_SOURCES})

target_include_directories(neurosim_core PUBLIC
    cpp/include
    ${HDF5_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(neurosim_core PUBLIC
    ${HDF5_CXX_LIBRARIES}
    Eigen3::Eigen
)

if(ENABLE_MPI)
    target_link_libraries(neurosim_core PUBLIC MPI::MPI_CXX)
endif()

if(ENABLE_OPENMP)
    target_link_libraries(neurosim_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ─────────────────────────────────────────────────────────────
#  Python bindings
# ─────────────────────────────────────────────────────────────
if(BUILD_PYTHON)
    pybind11_add_module(_neurosim_cpp
        cpp/src/python_bindings.cpp
    )
    target_link_libraries(_neurosim_cpp PRIVATE neurosim_core)
    set_target_properties(_neurosim_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/neurosim
    )
endif()

# ─────────────────────────────────────────────────────────────
#  Benchmark executable
# ─────────────────────────────────────────────────────────────
add_executable(neurosim_bench cpp/src/benchmark.cpp)
target_link_libraries(neurosim_bench PRIVATE neurosim_core)

# ─────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests/cpp)
endif()

# ─────────────────────────────────────────────────────────────
#  Install targets
# ─────────────────────────────────────────────────────────────
install(TARGETS neurosim_core neurosim_bench
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
)
install(DIRECTORY cpp/include/ DESTINATION include/neurosim)

# ─────────────────────────────────────────────────────────────
#  Version info
# ─────────────────────────────────────────────────────────────
configure_file(
    cmake/version.h.in
    ${CMAKE_BINARY_DIR}/include/neurosim/version.h
)

message(STATUS "NeuroSim CMake Configuration:")
message(STATUS "  CUDA:    ${ENABLE_CUDA}")
message(STATUS "  MPI:     ${ENABLE_MPI}")
message(STATUS "  OpenMP:  ${ENABLE_OPENMP}")
message(STATUS "  Python:  ${BUILD_PYTHON}")
message(STATUS "  Tests:   ${BUILD_TESTS}")
