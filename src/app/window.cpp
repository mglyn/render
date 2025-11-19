#include <iostream>

#include "window.h"

Window::~Window() {
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

bool Window::create(const Config& cfg) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    m_width = cfg.width;
    m_height = cfg.height;
    m_baseTitle = cfg.title;
    // Ensure the file is encoded in UTF-8
    m_window = glfwCreateWindow(m_width, m_height, m_baseTitle.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(cfg.vsync ? 1 : 0);
    m_prevTime = glfwGetTime();
    m_lastFpsUpdate = m_prevTime;
    return true;
}

void Window::beginFrame() {
    m_time = glfwGetTime();
    m_deltaTime = static_cast<float>(m_time - m_prevTime);
    m_prevTime = m_time;
    ++m_frameCount;
    updateFPS();
}
void Window::endFrame() { glfwSwapBuffers(m_window); }
bool Window::shouldClose() const { return m_window && glfwWindowShouldClose(m_window); }
void Window::pollEvents() { 
    // 在处理事件前，更新上一帧的按键状态
    // 这里我们只关心在main.cpp中用到的按键
    m_lastKeyState[GLFW_KEY_T] = isKeyDown(GLFW_KEY_T);
    // 如果有更多按键需要单次触发检测，在这里添加
    
    glfwPollEvents(); 
}

bool Window::isKeyJustPressed(int key) {
    bool currentState = isKeyDown(key);
    bool previousState = m_lastKeyState.count(key) ? m_lastKeyState[key] : false;
    return currentState && !previousState;
}

bool Window::isKeyDown(int key) const { return m_window && glfwGetKey(m_window, key) == GLFW_PRESS; }
void Window::updateFPS() {
    double now = m_time;
    if (now - m_lastFpsUpdate >= 1.0) {
        m_fps = m_frameCount / (now - m_lastFpsUpdate);
        m_lastFpsUpdate = now;
        m_frameCount = 0;
    }
}
