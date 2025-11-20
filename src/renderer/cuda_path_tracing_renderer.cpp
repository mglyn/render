#include "renderer/cuda_path_tracing_renderer.h"
#include "renderer/path_tracer.h"
#include "app/gpu_resources.h"
#include "scene/scene.h"
#include "scene/camera.h"
#include <iostream>
#include <cuda_runtime.h>

bool CudaPathTracingRenderer::init(int width, int height, GPUResources* resources, Scene* scene) {
    _width = width;
    _height = height;
    _gpu = resources;
    _scene = scene;

    // Allocate accumulation buffer
    cudaError_t err = cudaMalloc(&_d_accumulated_radiance, width * height * sizeof(glm::vec3));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate accumulation buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    cudaMemset(_d_accumulated_radiance, 0, width * height * sizeof(glm::vec3));

    _frame_count = 0;
    std::cout << "CudaPathTracingRenderer initialized." << std::endl;
    return true;
}

void CudaPathTracingRenderer::renderFrame(Camera& camera) {
    if (camera.isDirty() || _scene->isDirty()) {
        _frame_count = 0;
        cudaMemset(_d_accumulated_radiance, 0, _width * _height * sizeof(glm::vec3));
        camera.clearDirty();
        
        // If scene is dirty, rebuild and upload all data
        if (_scene->isDirty()) {
            _scene->buildAndUploadScene();
        }
    }

    launchKernel(camera);
    _frame_count++;
}

void CudaPathTracingRenderer::launchKernel(const Camera& camera) {
    // Map PBO resource
    uint8_t* pbo_ptr = nullptr;
    size_t num_bytes = 0;
    if (!_gpu->mapWrite(reinterpret_cast<void**>(&pbo_ptr), &num_bytes)) {
        return; // Failed to map
    }

    // Get scene data from the scene object
    const TriangleGPU* triangles = _scene->getTrianglesGPU();
    const MaterialGpu* materials = _scene->getMaterialsGPU();
    const BVHNodeGPU* bvhNodes = _scene->getBvhNodesGPU();
    const int* triIndices = _scene->getTriangleIndicesGPU();
    const int* lightIndices = _scene->getLightIndicesGPU();
    int triangleCount = _scene->getTriangleCount();
    int lightCount = _scene->getLightCount();

    // Call the kernel
    kernel_path_tracer(
        pbo_ptr, _width, _height,
        camera.getPosition(), camera.getViewMatrix(), camera.getFov(),
        _d_accumulated_radiance, _frame_count,
        triangles, triangleCount,
        materials,
        bvhNodes,
        triIndices,
        lightIndices, lightCount,
        maxDepth, samplesPerPixel, enableRussianRoulette, rouletteStartDepth,
        enableDiffuseImportanceSampling, lightingMode
    );

    // Unmap PBO resource
    _gpu->unmapWrite();
    _gpu->finalizeWrite();
}

void CudaPathTracingRenderer::destroy() {
    cudaFree(_d_accumulated_radiance);
    _d_accumulated_radiance = nullptr;
    // Scene GPU data is now managed by the Scene class
    std::cout << "CudaPathTracingRenderer destroyed." << std::endl;
}
