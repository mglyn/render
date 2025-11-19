#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include <string>
#include "scene/camera.h"
#include "app/gpu_resources.h"
#include "struct/shape.h"
#include "scene/scene.h"
#include "renderer/path_tracer.h"

// CUDA 路径追踪渲染器（原始实现迁移）
class CudaPathTracingRenderer {
public:

    bool init(int width, int height, GPUResources* gpu, Scene* scene);

    void renderFrame(Camera& camera);

    void destroy();

    const char* name() const { return "CudaPathTracing"; }

private:

    int _width = 0;
    int _height = 0;
    int _frame = 0;

    GPUResources* _gpu = nullptr;
    int _accumFrameCount = 0;
    glm::vec3* _accumBufferDev = nullptr;
    Scene* scene_ = nullptr;

    // GL objects
    GLuint _prog = 0; GLuint _vao=0; GLuint _vbo=0; GLuint _ebo=0;

    // models GPU data
    ModelGPU* _modelsDev = nullptr;
    int _modelCount = 0;
    IlluminantGPU* _illuminantsDev = nullptr;
    int _illuminantCount = 0;

    // Path tracing settings
    int _spp = 8;
    int _maxDepth = 5;

    bool _enableDiffuseImportanceSampling = true;
    LightingMode _lightingMode = LIGHTING_MODE_MIS;

public:
    int getSpp() const { return _spp; }
    void setSpp(int spp) { _spp = spp; resetAccumulation(); }
    
    int getMaxDepth() const { return _maxDepth; }
    void setMaxDepth(int depth) { _maxDepth = depth; resetAccumulation(); }

    bool getEnableDiffuseImportanceSampling() const { return _enableDiffuseImportanceSampling; }
    void setEnableDiffuseImportanceSampling(bool enable) { _enableDiffuseImportanceSampling = enable; resetAccumulation(); }
    
    LightingMode getLightingMode() const { return _lightingMode; }
    void setLightingMode(LightingMode mode) { _lightingMode = mode; resetAccumulation(); }

    bool initShaders();
    bool initQuad();
    std::string loadFile(const char* path);
    GLuint compileShader(GLenum type, const char* src);
    bool allocateAccumBuffer();
    void freeAccumBuffer();
    void resetAccumulation();
    bool uploadModels();
};
