#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "scene/camera.h"
#include "app/gpu_resources.h"
#include "struct/materialGpu.h"
#include "scene/scene.h"
#include "renderer/path_tracer.h"

// CUDA 路径追踪渲染器（原始实现迁移）
class CudaPathTracingRenderer {
public:
    CudaPathTracingRenderer() = default;
    ~CudaPathTracingRenderer() = default;

    bool init(int width, int height, GPUResources* resources, Scene* scene);
    void renderFrame(Camera& camera);
    void destroy();
    const char* name() const { return "CUDA Path Tracing Renderer"; }

    // UI-accessible parameters
    int maxDepth = 10;
    int samplesPerPixel = 1;
    bool enableRussianRoulette = true;
    int rouletteStartDepth = 3;
    bool enableDiffuseImportanceSampling = true;
    LightingMode lightingMode = LIGHTING_MODE_MIS;

private:
    void launchKernel(const Camera& camera);

    int _width = 0, _height = 0;
    GPUResources* _gpu = nullptr;
    Scene* _scene = nullptr;

    glm::vec3* _d_accumulated_radiance = nullptr;
    uint32_t _frame_count = 0;
};
