# CUDA + OpenGL 实时光线追踪示例

这个项目演示如何使用 CUDA 与 OpenGL 互操作 (通过 PBO) 实现一个最小的实时光线追踪渲染器。

要求 (Windows):
- Visual Studio (带 MSVC 工具链)
- CUDA Toolkit (与 Visual Studio 匹配)
- CMake >= 3.18

构建步骤 (PowerShell):

```powershell
mkdir build; cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

运行: `./Release/raytracer.exe`。

说明:
- 使用 GLFW 创建窗口，glad 作为 OpenGL 加载器（通过 CMake FetchContent 获取）。
- 使用 cudaGraphicsGLRegisterBuffer 注册 PBO，CUDA 内核将像素写入 PBO，然后通过 glTexSubImage2D 上传到纹理并显示。



