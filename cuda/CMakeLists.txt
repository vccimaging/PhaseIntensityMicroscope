cmake_minimum_required(VERSION 3.8)
project(CWS LANGUAGES C CXX CUDA)

# check requirements
if (UNIX)
	find_package(OpenCV 3.0 REQUIRED)
endif (UNIX)
if (WIN32)
	find_package(OpenCV 3.0 REQUIRED PATHS C:/Program\ Files/OpenCV)
endif (WIN32)

# set include directories
if (UNIX)
	set(CUDA_TARGET_INC ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include)
	set(CUDA_SAMPLE_INC ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
endif (UNIX)
if (WIN32)
	set(CUDA_SAMPLE_INC C:/ProgramData/NVIDIA\ Corporation/CUDA\ Samples/v10.0/common/inc)
endif (WIN32)
include_directories(deps src ${CUDA_TARGET_INC} ${CUDA_SAMPLE_INC} ${OpenCV_INCLUDE_DIRS})

# set link directories
if (UNIX)
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
endif (UNIX)

# build our static library 
set(STATIC_LIB_CWS_A_PHI cws_A_phi)
add_library(${STATIC_LIB_CWS_A_PHI} STATIC
  src/cws_A_phi.cu
  src/prepare_cufft_warmup.cu
)

# build cws
set(APP cws)
add_executable(${APP} cws.cpp ${CMAKE_SOURCE_DIR}/deps/argtable3/argtable3.c)
target_link_libraries(${APP} PRIVATE  ${STATIC_LIB_CWS_A_PHI} cufft ${OpenCV_LIBS})
install(TARGETS ${APP} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

# install
install(TARGETS ${STATIC_LIB_CWS_A_PHI}
        ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/lib
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src
        DESTINATION ${CMAKE_SOURCE_DIR}/include
        FILES_MATCHING PATTERN "*.h")
# NEED: Copy opencv_imgcodecs343.dll, opencv_imgproc343.dll and opencv_core343.dll to ${CMAKE_SOURCE_DIR}/bin, or make the folder contains these files be in PATH (as suggested by CMake warning).
