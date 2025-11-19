#pragma once

struct GLFWwindow;
class Scene;
class CudaPathTracingRenderer; // 来自 renderer 命名空间之外

namespace ui {

void init(GLFWwindow* window);
void shutdown();
void beginFrame();
void endFrame();

// 在这里添加你的UI组件声明
void renderUI(int& rendererMode, double fps, CudaPathTracingRenderer* renderer);
} // namespace ui
