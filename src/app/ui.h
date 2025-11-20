#pragma once

struct GLFWwindow;
class Scene;
class CudaPathTracingRenderer; // 来自 renderer 命名空间之外
class Camera;
class Application;

namespace ui {

void init(GLFWwindow* window);
void shutdown();
void beginFrame();
void endFrame();

// 在这里添加你的UI组件声明
void renderUI(int& rendererMode, 
    double fps, CudaPathTracingRenderer* renderer, 
    Camera& camera, Scene* scene, Application* app);
} // namespace ui
