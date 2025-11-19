#pragma once

#include <memory>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "window.h"
#include "scene/camera.h"
#include "app/gpu_resources.h"
#include "renderer/cuda_path_tracing_renderer.h"
#include "scene/scene.h"

class Application {
public:
    Application(int width, int height);
    ~Application();

    bool initialize();
    void run();
    void shutdown();

private:
    void createScene();
    void handleInput(float deltaTime);
    void render();
    void initRenderer(int mode);

    // Window and graphics
    int _width, _height;
    std::unique_ptr<Window> _window;
    std::unique_ptr<GPUResources> _gpu;

    // Scene and camera
    std::unique_ptr<Scene> _scene;
    std::unique_ptr<Camera> _camera;

    // Renderers
    std::unique_ptr<CudaPathTracingRenderer> _pathRenderer;
    CudaPathTracingRenderer* _currentRenderer = nullptr;
    int _currentMode;
    int _nextMode;

    // State
    bool _isRunning;
};
