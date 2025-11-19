cmake_minimum_required(VERSION 3.18)  
project(real_time_cuda_gl LANGUAGES CXX CUDA) 
set(CMAKE_CXX_STANDARD 20)             

# ============
# 查找OpenGL
# ============
find_package(OpenGL REQUIRED)          

# =====================
# 可执行文件与源码组织
# =====================
add_executable(raytracer                # 定义主可执行文件raytracer
  src/main.cpp                      # 主入口
  src/app/application.cpp               # 应用程序管理
  src/renderer/path_tracer.cu                # CUDA路径追踪核
  src/scene/camera.cpp                  # 相机实现
  src/scene/scene.cpp                   # 场景管理
  src/scene/model.cpp                   # 模型与BVH实现
  src/bvh/bvh.cpp
  src/renderer/intersect.cu               # 场景求交实现
  # Shape POD 定义在 shape.h（仅头文件）；后续如需单独加速结构可新增 .cu
  src/app/gpu_resources.cpp             # GPU资源管理
  src/app/window.cpp                    # 窗口管理
  src/renderer/cuda_path_tracing_renderer.cpp  # CUDA路径追踪渲染器
  src/app/ui.cpp                        # UI实现
)

target_include_directories(raytracer PRIVATE 
  ${CMAKE_SOURCE_DIR}/src              # 根src目录（头文件）
  ${CMAKE_SOURCE_DIR}/src/app          # app层
  ${CMAKE_SOURCE_DIR}/src/scene        # scene层
  ${CMAKE_SOURCE_DIR}/src/struct         
  ${CMAKE_SOURCE_DIR}/src/bvh
  ${CMAKE_SOURCE_DIR}/src/renderer)         
# =====================
# 第三方库路径设置
# =====================
set(GLFW_ROOT "${CMAKE_SOURCE_DIR}/thirdParty/glfw-3.3.8.bin.WIN64") # GLFW根目录
set(GLAD_ROOT "${CMAKE_SOURCE_DIR}/thirdParty/glad")                  # GLAD根目录
set(GLM_ROOT "${CMAKE_SOURCE_DIR}/thirdParty/glm-0.9.9.8")           # GLM根目录

# =============
# ImGui
# =============
add_subdirectory(thirdParty/imgui-1.92.4)

# =============
# GLFW 头文件
# =============
if(EXISTS "${GLFW_ROOT}/include/GLFW/glfw3.h")
  target_include_directories(raytracer PRIVATE "${GLFW_ROOT}/include")
endif()

# =============
# GLAD 头文件与源码
# =============
if(EXISTS "${GLAD_ROOT}/include/glad/glad.h")
  target_include_directories(raytracer PRIVATE "${GLAD_ROOT}/include")
  if(EXISTS "${GLAD_ROOT}/src/glad.c")
    message(STATUS "Including glad.c deterministically") # 明确包含glad.c源码
    target_sources(raytracer PRIVATE "${GLAD_ROOT}/src/glad.c")
    set_source_files_properties("${GLAD_ROOT}/src/glad.c" PROPERTIES LANGUAGE CXX)
  endif()
endif()

# =============
# GLM 头文件
# =============
if(EXISTS "${GLM_ROOT}/glm/glm.hpp")
  target_include_directories(raytracer PRIVATE "${GLM_ROOT}")
endif()

# =====================
# GLFW库链接与DLL拷贝
# =====================
set(GLFW_LIB "${GLFW_ROOT}/lib-vc2022/glfw3.lib") # 静态库路径
if(EXISTS "${GLFW_LIB}")
  target_link_libraries(raytracer PRIVATE "${GLFW_LIB}" OpenGL::GL imgui_lib) # 链接GLFW, OpenGL和ImGui
  # 拷贝DLL到输出目录，保证运行时依赖
  file(GLOB GLFW_DLLS "${GLFW_ROOT}/lib-vc2022/*.dll" "${GLFW_ROOT}/*.dll")
  foreach(d ${GLFW_DLLS})
    add_custom_command(TARGET raytracer POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${d}" "$<TARGET_FILE_DIR:raytracer>" )
  endforeach()
else()
  message(FATAL_ERROR "GLFW library not found at ${GLFW_LIB}. 请将预编译包放到thirdParty目录下.")
endif()

# =====================
# CUDA分离编译设置
# =====================
set_target_properties(raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # 支持CUDA分离编译

# =====================
# MSVC编译选项（CUDA运行库选择）
# =====================
if(MSVC)
  target_compile_options(raytracer PRIVATE
    # 统一 MSVC 运行时：C++ 源文件 Debug /MDd, Release /MD
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/MDd>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:/MD>
    # CUDA 主机编译器运行时一致
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--compiler-options="/MDd"> 
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:--compiler-options="/MD"> 
    # UTF-8 源编码
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options="/utf-8">
    # NVCC 设备端警告屏蔽（GLM默认函数重复属性，half类型弃用警告）
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=3012>
  )
endif()

# =====================
# 拷贝着色器到输出目录
# =====================
add_custom_command(TARGET raytracer POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/src/shaders" "$<TARGET_FILE_DIR:raytracer>/shaders"
  COMMENT "拷贝shaders到输出目录")

# =====================
# 配置完成提示
# =====================
message(STATUS "CMake配置完成，第三方库(glfw/glad/glm)请放在thirdParty目录下。")
