# 3.26: first version with the CUDA::nvJitLink imported target used below
cmake_minimum_required(VERSION 3.26)
project(SimLOD)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

option(SimLOD_CreateCUDASymlinks "Create symbolic links instead of copying CUDA files." OFF)

# Target SimLOD

add_executable(${PROJECT_NAME}
	src/GLRenderer.cpp
    include/unsuck_platform_specific.cpp	
    modules/progressive_octree/LasLoader.cpp	
    modules/progressive_octree/SimlodLoader.cpp
	modules/progressive_octree/CrashHandler.cpp
	modules/progressive_octree/main_progressive_octree.cpp)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                                               VS_DEBUGGER_COMMAND           "$<TARGET_FILE:${PROJECT_NAME}>"
                                               VS_DEBUGGER_ENVIRONMENT       "PATH=%PATH%;${CMAKE_PREFIX_PATH}")
# Optimisation flags come from CMake's per-config defaults (/O2 or -O3 in
# Release, none in Debug). Single-config generators default to Release.
# Multi-config generators (Visual Studio) leave CMAKE_BUILD_TYPE empty.
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	message(STATUS "No CMAKE_BUILD_TYPE specified, defaulting to Release.")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif ()

# Release builds also emit a .pdb so the crash handler can name the function
# and line in its stack trace. Keep the .pdb next to SimLOD.exe when copying
# the build to another machine. /OPT:REF,ICF keep the exe as small as without.
if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/Zi>)
	target_link_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/DEBUG /OPT:REF /OPT:ICF>)
endif ()

target_include_directories(${PROJECT_NAME} PRIVATE
    include
	modules/CudaPrint
	modules/progressive_octree)

# Dependencies

## CUDA toolkit
find_package(CUDAToolkit 12.0 REQUIRED)   # nvJitLink arrived in CUDA 12.0
target_include_directories(${PROJECT_NAME} PRIVATE
	${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
	CUDA::cuda_driver
	CUDA::nvrtc
	# CudaModularProgram.h includes <nvJitLink.h> and calls the API, but
	# upstream never links the library. Added in CUDA 12.0; the CMake
	# imported target CUDA::nvJitLink exists from CMake 3.26.
	CUDA::nvJitLink)

## OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY})

## fmt 
add_subdirectory(libs/fmt)
target_link_libraries(${PROJECT_NAME} fmt::fmt)

## glew
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/glew/include)
target_sources(${PROJECT_NAME} PRIVATE
	libs/glew/glew.c)
# glew.c is compiled into the exe, so the header must not declare its symbols
# as dllimport; without this every GL entry point raised C4273 (~3,600 warnings)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLEW_STATIC)

## GLFW
include(cmake/glfw.cmake)
target_include_directories(${PROJECT_NAME} PRIVATE
	${glfw_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} glfw)

## glm
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/glm)

## imgui
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/imgui
	libs/imgui/backends)
target_sources(${PROJECT_NAME} PRIVATE
	libs/imgui/imgui.cpp
	libs/imgui/imgui_demo.cpp
	libs/imgui/imgui_draw.cpp
	libs/imgui/imgui_tables.cpp
	libs/imgui/imgui_widgets.cpp
	libs/imgui/backends/imgui_impl_glfw.cpp
	libs/imgui/backends/imgui_impl_opengl3.cpp)

## implot
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/implot)
target_sources(${PROJECT_NAME} PRIVATE
	libs/implot/implot_items.cpp
	libs/implot/implot.cpp)

## laszip
add_subdirectory(libs/laszip)
target_link_libraries(${PROJECT_NAME} laszip)
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/laszip)

## TODO: disable copying of cuda source files
## - developer builds should use the project root dir as the working directory and directly load from ./modules
## - public release builds should be shipped with the precompiled cuda programs (cached in temp/cuda_programs)
# Post-Build
if (SimLOD_CreateCUDASymlinks)
	execute_process(
		COMMAND ${CMAKE_COMMAND} -E create_symlink
			"${PROJECT_SOURCE_DIR}/CMakelists.txt"
			"${CMAKE_BINARY_DIR}/.symlinktest.txt"
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		RESULT_VARIABLE symlinksSupported)
	if (symlinksSupported)
		message(STATUS "Cannot create symbolic links - missing user privileges. Falling back to copying.")
	else ()
		execute_process(
				COMMAND ${CMAKE_COMMAND} -E remove
				"${CMAKE_BINARY_DIR}/.symlinktest.txt"
				WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	endif (symlinksSupported)
endif (SimLOD_CreateCUDASymlinks)

if (symlinksSupported OR NOT SimLOD_CreateCUDASymlinks)
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_directory
			"${PROJECT_SOURCE_DIR}/modules"
			"$<TARGET_FILE_DIR:${PROJECT_NAME}>/modules"
		COMMENT "POST BUILD: copying modules folder")
else ()
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E create_symlink
			"${PROJECT_SOURCE_DIR}/modules"
			"$<TARGET_FILE_DIR:${PROJECT_NAME}>/modules"
		COMMENT "POST BUILD: creating symlink for modules folder")
endif (symlinksSupported OR NOT SimLOD_CreateCUDASymlinks)

