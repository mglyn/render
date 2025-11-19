#pragma once
#include <glm/glm.hpp>

class Camera;
class ModelGPU;
struct Shape;

extern "C" void launchPathTracer(
    unsigned int* pbo,
    int width,
    int height,
    const Camera& camera,
    const Shape* shapes,
    int shapeCount,
    const ModelGPU* models,
    int modelCount,
    int samplesPerPixel,
    int maxDepth,
    glm::vec3* accumBuffer,
    int accumFrameCount,
    int frameIndex);