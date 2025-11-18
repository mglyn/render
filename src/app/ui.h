#pragma once

struct GLFWwindow;

namespace ui {

void init(GLFWwindow* window);
void shutdown();
void beginFrame();
void endFrame();

// 在这里添加你的UI组件声明
void renderUI(int& rendererMode);

} // namespace ui
