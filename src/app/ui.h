#pragma once

struct GLFWwindow;
class Scene;

namespace ui {

void init(GLFWwindow* window);
void shutdown();
void beginFrame();
void endFrame();

// 在这里添加你的UI组件声明
void renderUI(int& rendererMode);
void renderBVHDebugUI(const Scene* scene);

} // namespace ui
