 
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "scene/camera.h"
#include "gpu/gpu_resources.h"
#include "app/window.h"
#include "app/renderer_base.h"
#include "app/cuda_path_tracing_renderer.h"
#include "app/wireframe_renderer.h"

int width = 1600, height = 900;

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

    // 可切换不同渲染器：0=PathTracing 1=Normal 2=ZBuffer 3=Wireframe
    int mode = 1; // 简单用数字控制，后续可做键盘切换
    RendererBase* renderer = nullptr;
    CudaPathTracingRenderer pathRenderer;
    WireframeRenderer wfRenderer;
    switch(mode){
        case 0: renderer = &pathRenderer; break;
        case 1: renderer = &wfRenderer; break;
    }
    if(!renderer->init(width,height,&gpu)){
        std::cerr << "Renderer init failed: " << renderer->name() << std::endl;
    } else {
        std::cout << "Active Renderer: " << renderer->name() << std::endl;
    }

    while(!window.shouldClose()){
        window.pollEvents();
        window.beginFrame();
        float deltaTime = window.deltaTime();

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
