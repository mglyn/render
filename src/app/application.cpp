#include "app/application.h"
#include "app/ui.h"

#include <iostream>
#include <glm/glm.hpp>

Application::Application(int width, int height)
    : _width(width), _height(height), 
      _renderScale(0.5f), // 默认1/2分辨率
      _currentRenderer(nullptr),
      _currentMode(0), _nextMode(0), _isRunning(false) {
    updateRenderResolution();
}

void Application::updateRenderResolution() {
    _renderWidth = static_cast<int>(_width * _renderScale);
    _renderHeight = static_cast<int>(_height * _renderScale);
    
    // 确保最小分辨率
    _renderWidth = std::max(_renderWidth, 64);
    _renderHeight = std::max(_renderHeight, 64);
}

void Application::setRenderScale(float scale) {
    _renderScale = glm::clamp(scale, 0.1f, 1.0f);
    updateRenderResolution();
    
    // 重新初始化GPU资源和渲染器
    if (_gpu) {
        _gpu->destroyPBO();
        if (!_gpu->createTriplePBO(_renderWidth, _renderHeight)) {
            std::cerr << "Failed to recreate GPU resources with new resolution" << std::endl;
            return;
        }
    }
    
    if (_currentRenderer) {
        _currentRenderer->destroy();
        if (!_currentRenderer->init(_renderWidth, _renderHeight, _gpu.get(), _scene.get())) {
            std::cerr << "Failed to reinitialize renderer with new resolution" << std::endl;
        }
    }
}

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
    if (!_gpu->createTriplePBO(_renderWidth, _renderHeight)) {
        std::cerr << "Failed to create GPU resources (triple PBO)" << std::endl;
        return false;
    }

    // 创建并加载显示着色器
    _displayShader = std::make_unique<Shader>();
    if (!_displayShader->load("shaders/fullscreen.vert", "shaders/texture.frag")) {
        std::cerr << "Failed to load display shader" << std::endl;
        return false;
    }

    // 创建全屏四边形
    float quadVertices[] = { 
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &_quadVAO);
    glGenBuffers(1, &_quadVBO);
    glBindVertexArray(_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, _quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glBindVertexArray(0);

    // 创建相机
    _camera = std::make_unique<Camera>(glm::vec3(0.0f, 0.0f, 0.0f));

    // 创建场景
    _scene = std::make_unique<Scene>();
    createScene();

    // 初始化渲染器
    _pathRenderer = std::make_unique<CudaPathTracingRenderer>();
    _currentRenderer = _pathRenderer.get();
    if (!_currentRenderer->init(_renderWidth, _renderHeight, _gpu.get(), _scene.get())) {
        std::cerr << "Renderer init failed: " << _currentRenderer->name() << std::endl;
    }

    _isRunning = true;
    return true;
}

void Application::createScene() {

    _camera->setPosition(glm::vec3(0.0f, 0.0f, 2.0f));
    
    auto makeWallPlane = [&](const Material& mat) {
        auto wall = std::make_unique<Model>(mat);
        glm::vec3 v0(0.0f, -5, -5);
        glm::vec3 v1(0.0f, -5, 5);
        glm::vec3 v2(0.0f, 5, 5);
        glm::vec3 v3(0.0f, 5, -5);
        glm::vec3 normal(1.0f, 0.0f, 0.0f); // All walls initially face inward from the side
        glm::vec2 t0(0,0), t1(0,1), t2(1,1), t3(1,0);

        wall->addTriangle({v0, v2, v1, normal, normal, normal, t0, t2, t1, mat});
        wall->addTriangle({v0, v3, v2, normal, normal, normal, t0, t3, t2, mat});
        return wall;
    };

    // 红左墙
    Material redWallMat{glm::vec3(0.8f, 0.1f, 0.1f), 0.0f};
    _scene->addModel(makeWallPlane(redWallMat), glm::vec3(-1, 0.0f, 0.0f), glm::vec3(0.0f), glm::vec3(1.0f));
    // 绿右墙
    Material greenWallMat{glm::vec3(0.1f, 0.8f, 0.1f), 0.0f};
    _scene->addModel(makeWallPlane(greenWallMat), glm::vec3(1, 0.0f, 0.0f), glm::vec3(0.0f, 180.0f, 0.0f), glm::vec3(1.0f));
    // 背景墙
    Material backWallMat{glm::vec3(0.4f, 0.4f, 0.45f), 0.0f};
    _scene->addModel(makeWallPlane(backWallMat), glm::vec3(0.0f, 0.0f, -1), glm::vec3(0.0f, 90.0f, 0.0f), glm::vec3(1.0f));
    // 地板
    Material floorMat{glm::vec3(0.5f, 0.5f, 0.55f), 0.0f};
    _scene->addModel(makeWallPlane(floorMat), glm::vec3(0.0f, -1, 0.0f), glm::vec3(0.0f, 0.0f, 90.0f), glm::vec3(1.0f));
    // 天花板
    Material ceilingMat{glm::vec3(0.35f, 0.35f, 0.4f), 0.0f};
    _scene->addModel(makeWallPlane(ceilingMat), glm::vec3(0.0f, 1, 0.0f), glm::vec3(0.0f, 0.0f, -90.0f), glm::vec3(1.0f));

    // 加载 OBJ 模型
    Material dragonMaterial{glm::vec3(0.7f, 0.7f, 0.9f), 0.0f}; // 淡蓝色，轻微金属感
    auto dragonModel = std::make_unique<Model>(dragonMaterial);
    if (dragonModel->loadObj("../../model/simple_dragon.obj", dragonMaterial)) {
        _scene->addModel(std::move(dragonModel), 
            glm::vec3(0.0f, 0.0f, 0.0f), // 位置
            glm::vec3(0, 90, 0),              // 旋转
            glm::vec3(1.8f));             // 缩放
        std::cout << "Successfully loaded and added simple_dragon.obj" << std::endl;
    } else {
        std::cerr << "Failed to load simple_dragon.obj" << std::endl;
    }

    // 加载光源模型
    Material lightMaterial{glm::vec3(0.0f, 1.0f, 1.0f), 0.0f}; // 无金属感
    auto lightModel = std::make_unique<Model>(lightMaterial, glm::vec3(30.0f, 30.0f, 30.0f)); // 强发光
    if (lightModel->loadObj("../../model/sphere.obj", lightMaterial)) {
        _scene->addModel(std::move(lightModel), 
            glm::vec3(0.0f, 1.0f, 0.0f), // 位置
            glm::vec3(0, 0, 0),              // 旋转
            glm::vec3(0.1f));             // 缩放
        std::cout << "Successfully loaded and added light_sphere.obj" << std::endl;
    } else {
        std::cerr << "Failed to load light_sphere.obj" << std::endl;
    }
}

void Application::handleInput(float deltaTime) {
    _window->pollEvents();

    // 相机旋转输入
    bool rotLeft = _window->isKeyDown(GLFW_KEY_LEFT);
    bool rotRight = _window->isKeyDown(GLFW_KEY_RIGHT);
    bool rotUp = _window->isKeyDown(GLFW_KEY_UP);
    bool rotDown = _window->isKeyDown(GLFW_KEY_DOWN);
    if(rotLeft || rotRight || rotUp || rotDown)
        _camera->processRotationInput(rotLeft, rotRight, rotUp, rotDown, deltaTime);

    // 相机移动输入
    bool forward = _window->isKeyDown(GLFW_KEY_W);
    bool backward = _window->isKeyDown(GLFW_KEY_S);
    bool left = _window->isKeyDown(GLFW_KEY_A);
    bool rightKey = _window->isKeyDown(GLFW_KEY_D);
    bool upKey = _window->isKeyDown(GLFW_KEY_SPACE);
    bool downKey = (_window->isKeyDown(GLFW_KEY_LEFT_SHIFT) || _window->isKeyDown(GLFW_KEY_RIGHT_SHIFT));
    if(forward || backward || left || rightKey || upKey || downKey)
        _camera->processMovement(forward, backward, left, rightKey, upKey, downKey, deltaTime, true);

    // ESC 退出
    if (_window->isKeyDown(GLFW_KEY_ESCAPE)) {
        _isRunning = false;
    }
}

void Application::render() {
    _window->beginFrame();
    
    // 1. 运行CUDA内核进行计算
    _currentRenderer->renderFrame(*_camera);

    // 2. 将PBO中的数据更新到纹理
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint displayPBO = _gpu->getDisplayPBO();
    GLuint displayTex = _gpu->getDisplayTexture();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, displayPBO);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _renderWidth, _renderHeight, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 3. 绘制带有该纹理的全屏四边形
    _displayShader->use();
    glBindVertexArray(_quadVAO);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    // 4. 渲染UI
    ui::beginFrame();
    ui::renderUI(_nextMode, _window->fps(), _pathRenderer.get(), *_camera, _scene.get(), this);
    ui::endFrame();

    _window->endFrame();
}

void Application::initRenderer(int mode) {
    if (_currentRenderer != nullptr) {
        _currentRenderer->destroy();
    }
    _currentRenderer = _pathRenderer.get();
    if (!_currentRenderer->init(_renderWidth, _renderHeight, _gpu.get(), _scene.get())) {
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

    // 清理显示资源
    if (_displayShader) {
        _displayShader.reset();
    }
    if (_quadVAO) {
        glDeleteVertexArrays(1, &_quadVAO);
        _quadVAO = 0;
    }
    if (_quadVBO) {
        glDeleteBuffers(1, &_quadVBO);
        _quadVBO = 0;
    }

    ui::shutdown();
    if (_window) {
        _window.reset();
    }
}
