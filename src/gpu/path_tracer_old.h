#pragma once
#include <glm/glm.hpp>

#include "scene/shape.h"

class Camera;
extern "C" void launchPathTracer(
    unsigned int* pbo,
    int width,
    int height,
    const Camera& camera,
    const Shape* shapes,
    int shapeCount,
    int samplesPerPixel,
    int maxDepth,
    glm::vec3* accumBuffer,
    int accumFrameCount,
    int frameIndex);
