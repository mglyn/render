#include "app/application.h"
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
    _camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 0.0f));

    // 创建场景
    _scene = std::make_unique<Scene>();
    createScene();

    // 初始化渲染器
    _pathRenderer = std::make_unique<CudaPathTracingRenderer>();
    _currentRenderer = _pathRenderer.get();
    if (!_currentRenderer->init(_width, _height, _gpu.get(), _scene.get())) {
        std::cerr << "Renderer init failed: " << _currentRenderer->name() << std::endl;
    }

    _isRunning = true;
    return true;
}

void Application::createScene() {

    _camera->setPosition(glm::vec3(0.0f, 0.0f, 2.0f));
    
    // 加载 OBJ 模型
    MaterialPOD dragonMaterial{glm::vec3(0.7f, 0.7f, 0.9f), 0.1f}; // 淡蓝色，轻微金属感
    auto dragonModel = std::make_unique<Model>(dragonMaterial);
    if (dragonModel->loadObj("../../model/simple_dragon.obj", dragonMaterial)) {
        _scene->addModel(std::move(dragonModel), 
            glm::vec3(0.0f, 0.0f, 0.0f), // 位置
            glm::vec3(0, 90, 0),              // 旋转
            glm::vec3(1.8f));             // 缩放
        std::cout << "Successfully loaded and added simple_dragon.obj" << std::endl;
    } else {
        std::cerr << "Failed to load simple_dragon.obj" << std::endl;
        // 如果加载失败，添加一个测试球体
        _scene->addShape(Shape::make_sphere(glm::vec3(0.0f, 0.0f, 0.0f), 0.5f, 
            MaterialPOD{glm::vec3(0.7f, 0.7f, 0.9f), 0.1f}));
    }

    // 地面
    _scene->addShape(Shape::make_plane(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 
        MaterialPOD{glm::vec3(0.8f, 0.8f, 0.8f), 0.0f}));
    // 天花板
    _scene->addShape(Shape::make_plane(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), 
        MaterialPOD{glm::vec3(0.8f, 0.8f, 0.8f), 0.0f}));
    // 红色左墙
    _scene->addShape(Shape::make_plane(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), MaterialPOD{glm::vec3(0.8f, 0.1f, 0.1f), 0.0f}));
    // 绿色右墙，位于 x = +1，朝向 -x
    _scene->addShape(Shape::make_plane(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), MaterialPOD{glm::vec3(0.1f, 0.8f, 0.1f), 0.0f}));
    // 背景墙 位于 z = -1，朝向 +z
    _scene->addShape(Shape::make_plane(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 0.0f, 1.0f), MaterialPOD{glm::vec3(0.8f, 0.8f, 0.8f), 0.0f}));
    // 添加光源
    _scene->addShape(Shape::make_sphere(glm::vec3(0.3f, 0.8f, 0.3f), 0.1f, 
        MaterialPOD{glm::vec3(0.0f), 0.0f, glm::vec3(1.0f, 1.0f, 0.9f) * 8.0f}));

    // 添加一个参考球体
    _scene->addShape(Shape::make_sphere(glm::vec3(-1.5f, 0.5f, 0.0f), 0.3f, 
        MaterialPOD{glm::vec3(0.9f, 0.2f, 0.2f), 0.0f}));
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

void Application::render() {
    _window->beginFrame();
    ui::beginFrame();

    _currentRenderer->renderFrame(*_camera);

    // 渲染 UI
    ui::renderUI(_nextMode, _window->fps());
    // ui::renderBVHDebugUI(_scene.get());

    ui::endFrame();
    _window->endFrame();
}

void Application::initRenderer(int mode) {
    if (_currentRenderer != nullptr) {
        _currentRenderer->destroy();
    }
    _currentRenderer = _pathRenderer.get();
    if (!_currentRenderer->init(_width, _height, _gpu.get(), _scene.get())) {
        std::cerr << "Renderer init failed: " << _currentRenderer->name() << std::endl;
    }
}

void Application::run() {
    while (_isRunning && !_window->shouldClose()) {
        float deltaTime = _window->deltaTime();

        handleInput(deltaTime);
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
