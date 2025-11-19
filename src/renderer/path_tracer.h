#pragma once
#include <glm/glm.hpp>

class Camera;
class ModelGPU;
struct IlluminantGPU;

enum LightingMode : int {
    LIGHTING_MODE_DIRECT = 0,
    LIGHTING_MODE_INDIRECT = 1,
    LIGHTING_MODE_MIS = 2
};

extern "C" void launchPathTracer(
    unsigned int* pbo,
    int width,
    int height,
    const Camera& camera,
    const ModelGPU* models,
    int modelCount,
    const IlluminantGPU* illuminants,
    int illuminantCount,
    int samplesPerPixel,
    int maxDepth,
    LightingMode lightingMode,
    bool enableDiffuseImportanceSampling,
    glm::vec3* accumBuffer,
    int accumFrameCount,
    int frameIndex);