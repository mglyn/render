#pragma once

#include <memory>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "window.h"
#include "scene/camera.h"
#include "app/gpu_resources.h"
#include "app/shader.h" // 新增
#include "renderer/cuda_path_tracing_renderer.h"
#include "scene/scene.h"

class Application {
public:
    Application(int width, int height);
    ~Application();

    bool initialize();
    void run();
    void shutdown();

    Scene* getScene() { return _scene.get(); }
    void setRenderScale(float scale);
    
    // Getters for UI
    float getRenderScale() const { return _renderScale; }
    int getRenderWidth() const { return _renderWidth; }
    int getRenderHeight() const { return _renderHeight; }
    int getDisplayWidth() const { return _width; }
    int getDisplayHeight() const { return _height; }

private:
    void createScene();
    void handleInput(float deltaTime);
    void render();
    void initRenderer(int mode);
    void updateRenderResolution();

    // Window and graphics
    int _width, _height;        // 显示分辨率
    int _renderWidth, _renderHeight; // 渲染分辨率
    float _renderScale;         // 渲染缩放因子 (0.1 - 1.0)
    std::unique_ptr<Window> _window;
    std::unique_ptr<GPUResources> _gpu;

    // Scene and camera
    std::unique_ptr<Scene> _scene;
    std::unique_ptr<Camera> _camera;

    // 显示渲染结果所需
    std::unique_ptr<Shader> _displayShader;
    GLuint _quadVAO, _quadVBO;

    // Renderers
    std::unique_ptr<CudaPathTracingRenderer> _pathRenderer;
    
    CudaPathTracingRenderer* _currentRenderer = nullptr;
    int _currentMode;
    int _nextMode;

    // State
    bool _isRunning;
};
