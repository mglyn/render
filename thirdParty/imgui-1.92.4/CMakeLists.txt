# thirdParty/imgui-1.92.4/CMakeLists.txt

# 创建一个静态库用于封装ImGui的源文件
add_library(imgui_lib STATIC
    imgui.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
    backends/imgui_impl_glfw.cpp
    backends/imgui_impl_opengl3.cpp
)

# 将ImGui的头文件目录公开，以便链接到此库的其他目标可以使用
target_include_directories(imgui_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
)

# 链接GLFW库（通过add_subdirectory构建）
target_link_libraries(imgui_lib PUBLIC glfw)

# 查找并链接OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(imgui_lib PUBLIC OpenGL::GL)

# 确保ImGui库使用与主项目相同的C++标准
set_target_properties(imgui_lib PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD})
