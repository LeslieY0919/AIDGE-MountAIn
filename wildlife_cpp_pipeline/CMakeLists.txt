cmake_minimum_required(VERSION 3.22)
project(wildlife_cpp_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../export_cpp_detector")
set(CLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../export_cpp_classifier")

file(GLOB_RECURSE DET_SOURCES "${DET_DIR}/dnn/src/*.cpp")
file(GLOB_RECURSE CLS_SOURCES "${CLS_DIR}/dnn/src/*.cpp")

add_library(det_raw STATIC ${DET_SOURCES})
target_include_directories(det_raw PRIVATE "${DET_DIR}/dnn" "${DET_DIR}/dnn/include")
target_compile_options(det_raw PRIVATE -O3 -ffast-math -march=native)

add_library(cls_raw STATIC ${CLS_SOURCES})
target_include_directories(cls_raw PRIVATE "${CLS_DIR}/dnn" "${CLS_DIR}/dnn/include")
target_compile_options(cls_raw PRIVATE -O3 -ffast-math -march=native)

set(DET_PREFIXED "${CMAKE_CURRENT_BINARY_DIR}/libdet_prefixed.a")
set(CLS_PREFIXED "${CMAKE_CURRENT_BINARY_DIR}/libcls_prefixed.a")

add_custom_command(
    OUTPUT ${DET_PREFIXED}
    COMMAND ${CMAKE_NM} -g --defined-only $<TARGET_FILE:det_raw>
            | sed -nE "s/^[0-9a-fA-F]+[[:space:]]+[A-Z][[:space:]]+([^[:space:]]+)/\\1 det_\\1/p"
            | sort -u > ${CMAKE_CURRENT_BINARY_DIR}/det_syms.map
    COMMAND ${CMAKE_OBJCOPY} --redefine-syms=${CMAKE_CURRENT_BINARY_DIR}/det_syms.map
            $<TARGET_FILE:det_raw> ${DET_PREFIXED}
    DEPENDS det_raw
    VERBATIM
)

add_custom_command(
    OUTPUT ${CLS_PREFIXED}
    COMMAND ${CMAKE_NM} -g --defined-only $<TARGET_FILE:cls_raw>
            | sed -nE "s/^[0-9a-fA-F]+[[:space:]]+[A-Z][[:space:]]+([^[:space:]]+)/\\1 cls_\\1/p"
            | sort -u > ${CMAKE_CURRENT_BINARY_DIR}/cls_syms.map
    COMMAND ${CMAKE_OBJCOPY} --redefine-syms=${CMAKE_CURRENT_BINARY_DIR}/cls_syms.map
            $<TARGET_FILE:cls_raw> ${CLS_PREFIXED}
    DEPENDS cls_raw
    VERBATIM
)

add_custom_target(prefixed_models DEPENDS ${DET_PREFIXED} ${CLS_PREFIXED})

add_executable(wildlife_pipeline main.cpp)
add_dependencies(wildlife_pipeline prefixed_models)
target_include_directories(wildlife_pipeline SYSTEM PRIVATE "${DET_DIR}")
target_compile_options(wildlife_pipeline PRIVATE -Wall -Wextra -O3 -ffast-math -march=native)
target_link_libraries(wildlife_pipeline PRIVATE ${DET_PREFIXED} ${CLS_PREFIXED} m)
