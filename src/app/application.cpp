#include "application.h"
#include "app/ui.h"
#include <iostream>
#include <glm/glm.hpp>

Application::Application(int width, int height)
    : _width(width), _height(height), _currentRenderer(nullptr),
      _currentMode(0), _nextMode(0), _isRunning(false) {}

Application::~Application() {
    shutdown();
}

bool Application::initialize() {
    // 创建窗口
    _window = std::make_unique<Window>();
    Window::Config cfg;
    cfg.width = _width;
    cfg.height = _height;
    cfg.title = "CUDA-OpenGL Raytracer";
    cfg.vsync = false;

    if (!_window->create(cfg)) {
        std::cerr << "Failed to create window" << std::endl;
        return false;
    }

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD" << std::endl;
        return false;
    }

    // 初始化 UI
    ui::init(_window->handle());

    // 创建 GPU 资源
    _gpu = std::make_unique<GPUResources>();
    if (!_gpu->createTriplePBO(_width, _height)) {
        std::cerr << "Failed to create GPU resources (triple PBO)" << std::endl;
        return false;
    }

    // 创建相机
    _camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 3.0f));

    // 创建场景
    _scene = std::make_unique<Scene>();
    createScene();

    // 初始化渲染器
    _pathRenderer = std::make_unique<CudaPathTracingRenderer>();
    _wfRenderer = std::make_unique<WireframeRenderer>();
    initRenderer(_currentMode);

    _isRunning = true;
    return true;
}

void Application::createScene() {
    // Cornell Box
    // Red wall
    _scene->addShape(Shape::make_sphere(glm::vec3(-1001.f, 0.f, 0.f), 1000.f, 
        MaterialPOD{glm::vec3(0.65f, 0.05f, 0.05f), 0.0f}));
    // Green wall
    _scene->addShape(Shape::make_sphere(glm::vec3(1001.f, 0.f, 0.f), 1000.f, 
        MaterialPOD{glm::vec3(0.12f, 0.45f, 0.15f), 0.0f}));
    // Floor
    _scene->addShape(Shape::make_sphere(glm::vec3(0.f, -1001.f, 0.f), 1000.f, 
        MaterialPOD{glm::vec3(0.73f, 0.73f, 0.73f)}));
    // Ceiling
    _scene->addShape(Shape::make_sphere(glm::vec3(0.f, 1001.f, 0.f), 1000.f, 
        MaterialPOD{glm::vec3(0.73f, 0.73f, 0.73f)}));
    // Back wall
    _scene->addShape(Shape::make_sphere(glm::vec3(0.f, 0.f, -1001.f), 1000.f, 
        MaterialPOD{glm::vec3(0.73f, 0.73f, 0.73f)}));

    // Metal sphere
    _scene->addShape(Shape::make_sphere(glm::vec3(-0.4f, -0.7f, 0.5f), 0.3f, 
        MaterialPOD{glm::vec3(0.8f, 0.6f, 0.2f), 1.0f}));
    // Glass sphere
    _scene->addShape(Shape::make_sphere(glm::vec3(0.3f, -0.4f, -0.2f), 0.2f, 
        MaterialPOD{glm::vec3(1.0f, 1.0f, 1.0f), 1.0f, glm::vec3(0.0f), 1.5f}));
    // Diffuse sphere
    _scene->addShape(Shape::make_sphere(glm::vec3(0.6f, -0.7f, 0.f), 0.3f, 
        MaterialPOD{glm::vec3(0.1f, 0.2f, 0.5f)}));

    // Light source
    _scene->addShape(Shape::make_sphere(glm::vec3(0.0f, 0.9f, 0.0f), 0.1f, 
        MaterialPOD{glm::vec3(0.0f), 0.0f, glm::vec3(1.0f, 0.8f, 0.6f) * 5.0f}));
}

void Application::handleInput(float deltaTime) {
    _window->pollEvents();

    // 按 'T' 键添加随机球体
    if (_window->isKeyJustPressed(GLFW_KEY_T)) {
        float r = static_cast<float>(rand()) / RAND_MAX * 0.2f + 0.1f;
        glm::vec3 pos(
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f,
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 1.0f,
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f
        );
        glm::vec3 albedo(
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX,
            static_cast<float>(rand()) / RAND_MAX
        );
        float metallic = static_cast<float>(rand()) / RAND_MAX;
        _scene->addShape(Shape::make_sphere(pos, r, MaterialPOD{albedo, metallic}));
        std::cout << "Added a new sphere. Total shapes: " << _scene->getShapes().size() << std::endl;
    }

    // 相机旋转输入
    bool rotLeft = _window->isKeyDown(GLFW_KEY_LEFT);
    bool rotRight = _window->isKeyDown(GLFW_KEY_RIGHT);
    bool rotUp = _window->isKeyDown(GLFW_KEY_UP);
    bool rotDown = _window->isKeyDown(GLFW_KEY_DOWN);
    _camera->processRotationInput(rotLeft, rotRight, rotUp, rotDown, deltaTime);

    // 相机移动输入
    bool forward = _window->isKeyDown(GLFW_KEY_W);
    bool backward = _window->isKeyDown(GLFW_KEY_S);
    bool left = _window->isKeyDown(GLFW_KEY_A);
    bool rightKey = _window->isKeyDown(GLFW_KEY_D);
    bool upKey = _window->isKeyDown(GLFW_KEY_SPACE);
    bool downKey = (_window->isKeyDown(GLFW_KEY_LEFT_SHIFT) || _window->isKeyDown(GLFW_KEY_RIGHT_SHIFT));

    _camera->processMovement(forward, backward, left, rightKey, upKey, downKey, deltaTime, true);

    // ESC 退出
    if (_window->isKeyDown(GLFW_KEY_ESCAPE)) {
        _isRunning = false;
    }
}

void Application::update(float deltaTime) {
    // 检查渲染器模式是否改变
    if (_nextMode != _currentMode) {
        _currentMode = _nextMode;
        initRenderer(_currentMode);
    }
}

void Application::render() {
    _window->beginFrame();
    ui::beginFrame();

    _currentRenderer->renderFrame(*_camera);

    // 渲染 UI
    ui::renderUI(_nextMode);

    ui::endFrame();
    _window->endFrame();
}

void Application::initRenderer(int mode) {
    if (_currentRenderer != nullptr) {
        _currentRenderer->destroy();
    }

    switch (mode) {
        case 0: _currentRenderer = _pathRenderer.get(); break;
        case 1: _currentRenderer = _wfRenderer.get(); break;
        default: _currentRenderer = _pathRenderer.get(); break;
    }

    if (!_currentRenderer->init(_width, _height, _gpu.get(), _scene.get())) {
        std::cerr << "Renderer init failed: " << _currentRenderer->name() << std::endl;
    } else {
        std::cout << "Active Renderer: " << _currentRenderer->name() << std::endl;
    }
}

void Application::run() {
    while (_isRunning && !_window->shouldClose()) {
        float deltaTime = _window->deltaTime();

        handleInput(deltaTime);
        update(deltaTime);
        render();
    }
}

void Application::shutdown() {
    if (_currentRenderer) {
        _currentRenderer->destroy();
    }
    if (_gpu) {
        _gpu->destroyPBO();
    }
    ui::shutdown();
    if (_window) {
        _window.reset();
    }
}
