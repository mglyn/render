#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include "struct/ray.h"
#include <curand_kernel.h>

// Constants
#define kPi 3.141592653589793f

// Lighting modes for UI
enum LightingMode {
    LIGHTING_MODE_DIRECT = 0,
    LIGHTING_MODE_INDIRECT = 1,
    LIGHTING_MODE_MIS = 2
};

// Forward declarations for structs to avoid circular includes
struct HitRecord;
struct TriangleGPU;
struct BVHNodeGPU;
struct MaterialGpu;

// Device utility functions
__device__ void writePixel(uint8_t* pbo_ptr, int width, int x, int y, const glm::vec3& color);
__device__ Ray generateCameraRay(float u, float v, const glm::vec3& cam_pos, const glm::mat4& cam_view, float fov, int width, int height);
__device__ glm::vec3 environmentColor(const Ray& r);
__device__ glm::vec3 uniformSampleHemisphere(const glm::vec3& normal, curandState* seed);
__device__ glm::vec3 cosineSampleHemisphere(const glm::vec3& normal, curandState* seed);
__device__ float bsdfPdf(bool useCosineSampling, float cosTheta);
__device__ float powerHeuristic(float pdfA, float pdfB);

// PBR functions
__device__ glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0);
__device__ float ggxDistribution(float NdotH, float roughness);
__device__ float smithG1(float NdotV, float roughness);
__device__ float smithGeometry(float NdotV, float NdotL, float roughness);
__device__ glm::vec3 sampleGGX(const glm::vec3& n, float roughness, curandState* seed, glm::vec3& H, float& pdf);
__device__ float getRoughnessFromMetallic(float metallic);

// Scene intersection
__device__ bool intersectScene(
    const Ray& r, float tMin, float tMax, HitRecord& rec,
    const BVHNodeGPU* bvhNodes,
    const TriangleGPU* triangles,
    const int* triIndices,
    const MaterialGpu* materials
);
