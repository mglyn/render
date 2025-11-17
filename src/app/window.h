#pragma once
#include <GLFW/glfw3.h>
#include <string>
#include <unordered_map>

class Window {
public:
    struct Config {
        int width = 800;
        int height = 600;
        std::string title = "CUDA Ray Tracer";
        bool vsync = false;
    };

    Window() = default;
    ~Window();

    bool create(const Config& cfg);
    void beginFrame();
    void endFrame();
    bool shouldClose() const;
    void pollEvents();
    bool isKeyDown(int key) const;
    float deltaTime() const { return m_deltaTime; }
    double time() const { return m_time; }
    double fps() const { return m_fps; }
    GLFWwindow* handle() const { return m_window; }
    int width() const { return m_width; }
    int height() const { return m_height; }
private:
    void updateFPS();
    GLFWwindow* m_window = nullptr;
    int m_width = 0;
    int m_height = 0;
    std::string m_baseTitle;
    double m_prevTime = 0.0;
    double m_time = 0.0;
    float m_deltaTime = 0.0f;
    int m_frameCount = 0;
    double m_fps = 0.0;
    double m_lastFpsUpdate = 0.0;
};
