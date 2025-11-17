#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include "renderer_base.h"
#include "scene/camera.h"
#include "gpu/gpu_resources.h"
#include "scene/shape.h"
#include "scene/scene.h"

// CUDA 路径追踪渲染器（原始实现迁移）
class CudaPathTracingRenderer : public RendererBase {
public:
    bool init(int width, int height, GPUResources* gpu, Scene* scene) override;
    void renderFrame(Camera& camera) override;
    void destroy() override;
    const char* name() const override { return "CudaPathTracing"; }
private:
    int _width = 0;
    int _height = 0;
    int _frame = 0;
    GPUResources* _gpu = nullptr;
    int _accumFrameCount = 0;
    glm::vec3* _accumBufferDev = nullptr;
    Shape* _shapesDev = nullptr;
    int _shapeCount = 0;
    // camera state for accumulation reset
    bool _hasPrevCamState = false;
    glm::vec3 _prevCamPos{};
    glm::vec3 _prevCamFront{};
    glm::vec3 _prevCamUp{};
    float _prevFov = 0.0f;
    // GL objects
    GLuint _prog = 0; GLuint _vao=0; GLuint _vbo=0; GLuint _ebo=0;

    bool initShaders();
    bool initQuad();
    std::string loadFile(const char* path);
    GLuint compileShader(GLenum type, const char* src);
    bool allocateAccumBuffer();
    void freeAccumBuffer();
    void resetAccumulation();
    bool cameraChanged(const Camera& camera) const;
    void updateCameraState(const Camera& camera);
    bool uploadScene();
};
