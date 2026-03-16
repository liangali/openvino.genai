# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

add_executable(modeling_qwen3_vl
    modeling_qwen3_vl.cpp
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat/load_image.cpp"
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_qwen3_vl PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_qwen3_vl PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_qwen3_vl PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_qwen3_vl openvino_genai)

if(WIN32)
    set(tokenizers_dll_name "openvino_tokenizers$<$<CONFIG:Debug>:d>.dll")
elseif(APPLE)
    set(tokenizers_dll_name "libopenvino_tokenizers$<$<CONFIG:Debug>:d>.dylib")
else()
    set(tokenizers_dll_name "libopenvino_tokenizers$<$<CONFIG:Debug>:d>.so")
endif()

add_custom_command(TARGET modeling_qwen3_vl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_qwen3_vl>"
    VERBATIM)

set_target_properties(modeling_qwen3_vl PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

add_executable(modeling_qwen3_5
    modeling_qwen3_5.cpp
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat/load_image.cpp"
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_qwen3_5 PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_qwen3_5 PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_qwen3_5 PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_qwen3_5 openvino_genai)

add_custom_command(TARGET modeling_qwen3_5 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_qwen3_5>"
    VERBATIM)

set_target_properties(modeling_qwen3_5 PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

add_executable(modeling_deepseek_ocr2
    modeling_deepseek_ocr2.cpp
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat/load_image.cpp"
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_deepseek_ocr2 PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/visual_language_chat"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_deepseek_ocr2 PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_deepseek_ocr2 PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_deepseek_ocr2 openvino_genai)

add_custom_command(TARGET modeling_deepseek_ocr2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_deepseek_ocr2>"
    VERBATIM)

set_target_properties(modeling_deepseek_ocr2 PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

add_executable(modeling_zimage
    modeling_zimage.cpp
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/image_generation/imwrite.cpp"
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_zimage PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/image_generation"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_zimage PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_zimage PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_zimage openvino_genai)

add_custom_command(TARGET modeling_zimage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_zimage>"
    VERBATIM)

set_target_properties(modeling_zimage PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

add_executable(modeling_wan_t2v
    modeling_wan_t2v.cpp
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/image_generation/imwrite.cpp"
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_wan_t2v PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/samples/cpp/image_generation"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_wan_t2v PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_wan_t2v PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_wan_t2v openvino_genai)

add_custom_command(TARGET modeling_wan_t2v POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_wan_t2v>"
    VERBATIM)

set_target_properties(modeling_wan_t2v PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

add_executable(modeling_wan_layered_dit
    modeling_wan_layered_dit.cpp
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_wan_layered_dit PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_wan_layered_dit PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_wan_layered_dit PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_wan_layered_dit openvino_genai)

add_custom_command(TARGET modeling_wan_layered_dit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_wan_layered_dit>"
    VERBATIM)

set_target_properties(modeling_wan_layered_dit PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)
    
add_executable(modeling_dflash
    modeling_dflash.cpp
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_dflash PRIVATE
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_dflash PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_dflash PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_dflash openvino_genai)

add_custom_command(TARGET modeling_dflash POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_dflash>"
    VERBATIM)

set_target_properties(modeling_dflash PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

# Qwen3-TTS sample
add_executable(modeling_qwen3_tts
    modeling_qwen3_tts.cpp
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_qwen3_tts PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_qwen3_tts PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_qwen3_tts PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_qwen3_tts openvino_genai)

add_custom_command(TARGET modeling_qwen3_tts POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_qwen3_tts>"
    VERBATIM)

set_target_properties(modeling_qwen3_tts PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)

# Qwen3-ASR sample
add_executable(modeling_qwen3_asr
    modeling_qwen3_asr.cpp
    $<TARGET_OBJECTS:openvino_genai_obj>)

target_include_directories(modeling_qwen3_asr PRIVATE
    "${CMAKE_BINARY_DIR}"
    "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/src"
    "${OpenVINOGenAI_SOURCE_DIR}/openvino/thirdparty/json/nlohmann_json/include"
    $<TARGET_PROPERTY:openvino::genai,INTERFACE_INCLUDE_DIRECTORIES>)

target_link_libraries(modeling_qwen3_asr PRIVATE
    $<TARGET_PROPERTY:openvino_genai_obj,LINK_LIBRARIES>)

target_compile_definitions(modeling_qwen3_asr PRIVATE
    openvino_genai_EXPORTS)

add_dependencies(modeling_qwen3_asr openvino_genai)

add_custom_command(TARGET modeling_qwen3_asr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE_DIR:openvino_genai>/${tokenizers_dll_name}"
            "$<TARGET_FILE_DIR:modeling_qwen3_asr>"
    VERBATIM)

set_target_properties(modeling_qwen3_asr PROPERTIES
    # Ensure out of box LC_RPATH on macOS with SIP
    INSTALL_RPATH_USE_LINK_PATH ON)