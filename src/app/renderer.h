#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include "scene/camera.h"
#include "gpu/gpu_resources.h"
#include "scene/shape.h"

class Renderer {
public:

    bool init(int width, int height, GPUResources* gpu);
    void renderFrame(Camera& camera); // CUDA map->kernel->unmap, PBO->texture, fullscreen draw
    void destroy();

private:
    int _width = 0;
    int _height = 0;
    int _frame = 0;
    GPUResources* _gpu = nullptr;

    int _accumFrameCount = 0;
    glm::vec3* _accumBufferDev = nullptr;
    // 形状数据（上传到 GPU）
    Shape* _shapesDev = nullptr;
    int _shapeCount = 0;
    mutable bool _hasPrevCamState = false;
    mutable glm::vec3 _prevCamPos{};
    mutable glm::vec3 _prevCamFront{};
    mutable glm::vec3 _prevCamUp{};
    mutable float _prevFov = 0.0f;
    // GL objects
    GLuint _prog = 0;
    GLuint _vao = 0;
    GLuint _vbo = 0;
    GLuint _ebo = 0;

    bool allocateAccumBuffer();
    void freeAccumBuffer();
    void resetAccumulation();
    bool cameraChanged(const Camera& camera) const;
    void updateCameraState(const Camera& camera);

    // helpers
    bool initShaders();
    bool initQuad();
    std::string loadFile(const char* path);
    GLuint compileShader(GLenum type, const char* src);
};
