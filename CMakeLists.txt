cmake_minimum_required(VERSION 3.10)

# Set the project name
project(SdfTask CXX)

# Find the SDL2 library
find_package(SDL2 REQUIRED)
find_package(Vulkan REQUIRED)

# Include SDL2 headers
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/LiteMath/external)
include_directories(${SDL2_INCLUDE_DIRS})

add_compile_definitions(USE_STB_IMAGE)

# Add the executable
add_executable(render
    application.cpp
    device.cpp
    main.cpp
    mesh.cpp
    pipeline.cpp
    window.cpp
    grid.cpp
    octree.cpp)

file(GLOB_RECURSE GLSL_SOURCE_FILES CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/shaders/*.frag"
        "${PROJECT_SOURCE_DIR}/shaders/*.vert"
        "${PROJECT_SOURCE_DIR}/shaders/*.comp"
)

foreach (GLSL IN LISTS GLSL_SOURCE_FILES)
    file(RELATIVE_PATH FILE_NAME "${PROJECT_SOURCE_DIR}/shaders/" "${GLSL}")
    set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")

    add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND Vulkan::glslc ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL})

    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach (GLSL)

add_custom_target(
        ShadersTarget ALL
        DEPENDS ${SPIRV_BINARY_FILES}
)

# Link the SDL2 library to the executable
target_link_libraries(render ${SDL2_LIBRARIES} Vulkan::Vulkan)

# Set path to executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
