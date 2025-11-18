#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "scene/camera.h"
#include "gpu/gpu_resources.h"
#include "app/window.h"
#include "renderer/renderer_base.h"
#include "renderer/cuda_path_tracing_renderer.h"
#include "renderer/wireframe_renderer.h"
#include "scene/scene.h"

int width = 1600, height = 900;

// 在main函数外创建场景
void createScene(Scene& scene) {
    // Cornell Box
    // Red wall
    scene.addShape(Shape::make_sphere(glm::vec3(-1001.f, 0.f, 0.f), 1000.f, MaterialPOD{glm::vec3(0.65f, 0.05f, 0.05f), 0.0f}));
    // Green wall
    scene.addShape(Shape::make_sphere(glm::vec3(1001.f, 0.f, 0.f), 1000.f, MaterialPOD{glm::vec3(0.12f, 0.45f, 0.15f), 0.0f}));
    // Floor
    scene.addShape(Shape::make_sphere(glm::vec3(0.f, -1001.f, 0.f), 1000.f, MaterialPOD{glm::vec3(0.73f, 0.73f, 0.73f)}));
    // Ceiling
    scene.addShape(Shape::make_sphere(glm::vec3(0.f, 1001.f, 0.f), 1000.f, MaterialPOD{glm::vec3(0.73f, 0.73f, 0.73f)}));
    // Back wall
    scene.addShape(Shape::make_sphere(glm::vec3(0.f, 0.f, -1001.f), 1000.f, MaterialPOD{glm::vec3(0.73f, 0.73f, 0.73f)}));

    // Metal sphere
    scene.addShape(Shape::make_sphere(glm::vec3(-0.4f, -0.7f, 0.5f), 0.3f, MaterialPOD{glm::vec3(0.8f, 0.6f, 0.2f), 1.0f}));
    // Glass sphere
    scene.addShape(Shape::make_sphere(glm::vec3(0.3f, -0.4f, -0.2f), 0.2f, MaterialPOD{glm::vec3(1.0f, 1.0f, 1.0f), 1.0f, glm::vec3(0.0f), 1.5f}));
    // Diffuse sphere
    scene.addShape(Shape::make_sphere(glm::vec3(0.6f, -0.7f, 0.f), 0.3f, MaterialPOD{glm::vec3(0.1f, 0.2f, 0.5f)}));

    // Light source
    scene.addShape(Shape::make_sphere(glm::vec3(0.0f, 0.9f, 0.0f), 0.1f, MaterialPOD{glm::vec3(0.0f), 0.0f, glm::vec3(1.0f, 0.8f, 0.6f) * 5.0f}));

    // Add a triangle for wireframe testing
    // scene.addShape(Shape::make_triangle(
    //     glm::vec3(-0.5f, -0.5f, 1.5f),
    //     glm::vec3(0.5f, -0.5f, 1.5f),
    //     glm::vec3(0.0f, 0.5f, 1.5f),
    //     MaterialPOD{glm::vec3(1.0f, 0.0f, 0.0f)}
    // ));
}

int main(){
    Window::Config cfg; 
    cfg.width = width; 
    cfg.height = height; 
    cfg.title = "CUDA-OpenGL Raytracer"; 
    cfg.vsync = false;
    Window window;
    if(!window.create(cfg)) 
        return -1;

    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr<<"Failed to init GLAD"<<std::endl; return -1;
    }

    Camera camera(glm::vec3(0.0f,0.0f,3.0f));
    GPUResources gpu;
    if(!gpu.createTriplePBO(width, height)){
        std::cerr<<"Failed to create GPU resources (triple PBO)"<<std::endl;
    }

    // 创建并填充场景
    Scene scene;
    createScene(scene);

    // 可切换不同渲染器：0=PathTracing 1=Wireframe
    int mode = 0; // 切换到线框模式进行测试
    RendererBase* renderer = nullptr;
    CudaPathTracingRenderer pathRenderer;
    WireframeRenderer wfRenderer;
    switch(mode){
        case 0: renderer = &pathRenderer; break;
        case 1: renderer = &wfRenderer; break;
    }
    if(!renderer->init(width,height,&gpu, &scene)){
        std::cerr << "Renderer init failed: " << renderer->name() << std::endl;
    } else {
        std::cout << "Active Renderer: " << renderer->name() << std::endl;
    }

    while(!window.shouldClose()){
        window.pollEvents();
        window.beginFrame();
        float deltaTime = window.deltaTime();

        // 示例：按 'T' 键添加一个随机球体来测试场景更新
        if (window.isKeyJustPressed(GLFW_KEY_T)) {
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
            scene.addShape(Shape::make_sphere(pos, r, MaterialPOD{albedo, metallic}));
            std::cout << "Added a new sphere. Total shapes: " << scene.getShapes().size() << std::endl;
        }

        // 视角旋转输入与移动输入统一处理
        bool rotLeft = window.isKeyDown(GLFW_KEY_LEFT);
        bool rotRight = window.isKeyDown(GLFW_KEY_RIGHT);
        bool rotUp = window.isKeyDown(GLFW_KEY_UP);
        bool rotDown = window.isKeyDown(GLFW_KEY_DOWN);
        camera.processRotationInput(rotLeft, rotRight, rotUp, rotDown, deltaTime);

        bool forward = window.isKeyDown(GLFW_KEY_W);
        bool backward = window.isKeyDown(GLFW_KEY_S);
        bool left = window.isKeyDown(GLFW_KEY_A);
        bool rightKey = window.isKeyDown(GLFW_KEY_D);
        bool upKey = window.isKeyDown(GLFW_KEY_SPACE);
        bool downKey = (window.isKeyDown(GLFW_KEY_LEFT_SHIFT) || window.isKeyDown(GLFW_KEY_RIGHT_SHIFT));
        if(window.isKeyDown(GLFW_KEY_ESCAPE)) glfwSetWindowShouldClose(window.handle(), 1);
        camera.processMovement(forward, backward, left, rightKey, upKey, downKey, deltaTime, true);

        renderer->renderFrame(camera);
        window.endFrame();
    }

    renderer->destroy();
    gpu.destroyPBO();
}
