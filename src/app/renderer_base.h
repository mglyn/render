#pragma once
#include <string>

class Camera;
class GPUResources;
class Scene; // 前向声明

// 抽象渲染器基类：提供统一接口，便于扩展不同渲染实现
class RendererBase {
public:
    virtual ~RendererBase() = default;
    virtual bool init(int width, int height, GPUResources* gpu, Scene* scene) = 0;
    virtual void renderFrame(Camera& camera) = 0;
    virtual void destroy() = 0;
    virtual const char* name() const = 0;

protected:
    Scene* scene_ = nullptr; // 存储场景指针
};
