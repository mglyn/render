#include "path_tracer.h"
#include "scene/camera.h"
#include "struct/shape.h"
#include "struct/ray.h"
#include "renderer/pt_utils.h"
#include "renderer/intersect.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <vector>
#include <float.h>
#include <math.h>
#include <optional>

// =========================
// 路径追踪 (CUDA)
// =========================
// 包含球体相交、场景遍历、Fresnel概率混合采样、直接光照、环境光、累积采样等

namespace {
    // 天空环境光颜色（渐变）
    __device__ glm::vec3 environmentColor(const Ray &r) {
        glm::vec3 unitDir = glm::normalize(r.direction());
        float t = 0.5f * (unitDir.y + 1.0f);
        return (1.0f - t) * glm::vec3(0.3f, 0.3f, 0.3f) + t * glm::vec3(0.35f, 0.35f, 0.5);
    }

    constexpr float kPi = 3.1415926535f;

    __device__ float powerHeuristic(float pdfA, float pdfB) {
        float a = pdfA * pdfA;
        float b = pdfB * pdfB;
        float denom = a + b;
        if (denom <= 0.0f) return 0.0f;
        return a / denom;
    }

    __device__ float lightPdfFromHit(
        const IlluminantGPU* illuminants,
        int illuminantCount,
        int lightIndex,
        int triangleIndex,
        const glm::vec3& rayOrigin,
        const glm::vec3& hitPoint,
        const glm::vec3& lightNormal)
    {
        if (!illuminants || lightIndex < 0 || lightIndex >= illuminantCount) return 0.0f;
        const IlluminantGPU& light = illuminants[lightIndex];
        if (!light.triangles || triangleIndex < 0 || triangleIndex >= light.triangleCount) return 0.0f;

        const TriangleGPU& tri = light.triangles[triangleIndex];
        glm::vec3 v0 = tri.v0;
        glm::vec3 v1 = tri.v1;
        glm::vec3 v2 = tri.v2;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        float area = 0.5f * glm::length(glm::cross(edge1, edge2));
        if (area <= 1e-6f) return 0.0f;

        float pdfArea = 1.0f / (static_cast<float>(illuminantCount) * static_cast<float>(light.triangleCount) * area);
        glm::vec3 toLight = hitPoint - rayOrigin;
        float dist2 = glm::dot(toLight, toLight);
        if (dist2 <= 1e-8f) return 0.0f;
        float dist = sqrtf(dist2);
        glm::vec3 wi = toLight / dist;
        float cosLight = fmaxf(glm::dot(lightNormal, -wi), 0.0f);
        if (cosLight <= 1e-6f) return 0.0f;

        return pdfArea * dist2 / cosLight;
    }

    __device__ bool sampleIlluminantSurface(
        const IlluminantGPU* illuminants,
        int illuminantCount,
        uint32_t& seed,
        glm::vec3& lightPos,
        glm::vec3& lightNormal,
        glm::vec3& lightEmission,
        float& pdfArea)
    {
        if (!illuminants || illuminantCount <= 0) return false;

        int lightIdx = static_cast<int>(rand01(seed) * illuminantCount);
        lightIdx = glm::clamp(lightIdx, 0, illuminantCount - 1);
        const IlluminantGPU& light = illuminants[lightIdx];
        if (light.triangleCount <= 0 || !light.triangles) return false;

        int triIdx = static_cast<int>(rand01(seed) * light.triangleCount);
        triIdx = glm::clamp(triIdx, 0, light.triangleCount - 1);
        const TriangleGPU& tri = light.triangles[triIdx];

        glm::vec3 v0 = tri.v0;
        glm::vec3 v1 = tri.v1;
        glm::vec3 v2 = tri.v2;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 triNormal = glm::cross(edge1, edge2);
        float area = 0.5f * glm::length(triNormal);
        if (area <= 1e-6f) return false;
        lightNormal = glm::normalize(triNormal);

        float u1 = rand01(seed);
        float u2 = rand01(seed);
        float su1 = sqrtf(u1);
        float baryA = 1.0f - su1;
        float baryB = u2 * su1;
        float baryC = 1.0f - baryA - baryB;
        lightPos = baryA * v0 + baryB * v1 + baryC * v2;

        lightEmission = light.emission;
        pdfArea = 1.0f / (static_cast<float>(illuminantCount) * static_cast<float>(light.triangleCount) * area);
        return true;
    }

    __device__ float bsdfPdf(bool importanceSampling, float cosTheta){
        if (importanceSampling) {
            return cosTheta > 0.0f ? cosTheta / kPi : 0.0f;
        }
        return 1.0f / (2.0f * kPi);
    }

    __device__ glm::vec3 sampleDirectLightingContribution(
        const ModelGPU* models,
        int modelCount,
        const IlluminantGPU* illuminants,
        int illuminantCount,
        uint32_t& seed,
        const glm::vec3& point,
        const glm::vec3& normal,
        const glm::vec3& brdf,
        const glm::vec3& baseWeight,
        float pDiff,
        bool enableDiffuseImportanceSampling,
        LightingMode lightingMode)
    {
        if (!illuminants || illuminantCount <= 0 || glm::length(brdf) <= 0.0f || glm::length(baseWeight) <= 0.0f) {
            return glm::vec3(0.0f);
        }

        glm::vec3 lightPos;
        glm::vec3 lightNormal;
        glm::vec3 lightEmission;
        float pdfArea = 0.0f;
        if (!sampleIlluminantSurface(illuminants, illuminantCount, seed, lightPos, lightNormal, lightEmission, pdfArea)) {
            return glm::vec3(0.0f);
        }

        glm::vec3 toLight = lightPos - point;
        float dist2 = glm::dot(toLight, toLight);
        if (dist2 <= 1e-6f) {
            return glm::vec3(0.0f);
        }
        float dist = sqrtf(dist2);
        glm::vec3 wi = toLight / dist;
        float cosThetaSurface = fmaxf(glm::dot(normal, wi), 0.0f);
        float cosThetaLight = fmaxf(glm::dot(lightNormal, -wi), 0.0f);
        if (cosThetaSurface <= 0.0f || cosThetaLight <= 0.0f) {
            return glm::vec3(0.0f);
        }

        float pdfDir = pdfArea * dist2 / cosThetaLight;
        if (pdfDir <= 1e-6f) {
            return glm::vec3(0.0f);
        }

        float weightNEE = 1.0f;
        if (lightingMode == LIGHTING_MODE_MIS && pDiff > 0.0f) {
            float pdfLocal = bsdfPdf(enableDiffuseImportanceSampling, cosThetaSurface);
            float pdfBsdfLightDir = pDiff * pdfLocal;
            weightNEE = powerHeuristic(pdfDir, pdfBsdfLightDir);
        }

        bool occluded = false;
        float maxT = dist - 0.001f;
        if (maxT > 0.0f) {
            Ray shadowRay(point + normal * 0.001f, wi);
            if (modelCount > 0 && models) {
                occluded = anyHit(models, modelCount, shadowRay, 0.001f, maxT);
            }
            if (!occluded && illuminantCount > 0 && illuminants) {
                occluded = anyHit(illuminants, illuminantCount, shadowRay, 0.001f, maxT);
            }
        }

        if (occluded) {
            return glm::vec3(0.0f);
        }

        return baseWeight * brdf * lightEmission * (cosThetaSurface / pdfDir) * weightNEE;
    }

    // 路径追踪主循环：Fresnel概率混合采样，累积throughput
    __device__ glm::vec3 traceRay(const ModelGPU* models, int modelCount,
                                  const IlluminantGPU* illuminants, int illuminantCount,
                                  Ray r, uint32_t &seed, int maxDepth,
                                  LightingMode lightingMode,
                                  bool enableDiffuseImportanceSampling,
                                  bool enableRussianRoulette,
                                  int rouletteStartDepth) 
    {
        glm::vec3 radiance(0.0f);
        glm::vec3 throughput(1.0f);
        float prevBsdfPdf = 0.0f;
        bool prevBounceDiffuse = false;

        for (int depth = 0; depth < maxDepth; ++depth) {
            // 统一求交：同时检查BVH和基础形状，选择最近命中
            HitRecord rec;
            bool hit = false;
            float closest = FLT_MAX;
            glm::vec3 rayOrigin = r.origin();
            
            // 检查模型（保持最近命中）
            if (modelCount > 0 && models) {
                HitRecord bvhRec;
                if (intersectModels(models, modelCount, r, 0.001f, closest, bvhRec)) {
                    hit = true;
                    closest = bvhRec.t;
                    rec = bvhRec;
                }
            }

            // 检查发光体（用当前 closest 作为上界，仅更近时覆盖）
            if (illuminantCount > 0 && illuminants) {
                HitRecord bvhRec;
                if (intersectModels(illuminants, illuminantCount, r, 0.001f, closest, bvhRec)) {
                    hit = true;
                    closest = bvhRec.t;
                    rec = bvhRec;
                }
            }
            
            // 未命中返回环境光
            if (!hit) {
                radiance += throughput * environmentColor(r);
                break;
            }

            // 自发光累加（MIS 对命中光源时的权重修正）
            if (glm::dot(rec.emission, rec.emission) > 0.0f) {
                float weight = 1.0f;
                if (lightingMode == LIGHTING_MODE_MIS && prevBounceDiffuse && prevBsdfPdf > 0.0f && rec.fromIlluminant) {
                    float pdfLight = lightPdfFromHit(illuminants, illuminantCount, rec.objectIndex, rec.primitiveIndex, rayOrigin, rec.point, rec.normal);
                    if (pdfLight > 0.0f) {
                        weight = powerHeuristic(prevBsdfPdf, pdfLight);
                    } else {
                        weight = 1.0f;
                    }
                }
                radiance += throughput * rec.emission * weight;
            }

            glm::vec3 throughputPrev = throughput;
            float currBsdfPdf = 0.0f;
            bool currBounceDiffuse = false;

            // 基于金属度的离散混合采样。
            // 为了保持无偏，需要对选择的采样策略按其选择概率进行重要性权重校正。
            float pSpec = rec.metallic;
            float pDiff = 1.0f - pSpec;
            float choice = rand01(seed);
            if (choice < pSpec) {
                // 镜面反射分支（delta分布）。
                if (lightingMode == LIGHTING_MODE_DIRECT) {
                    break;
                }
                glm::vec3 refl = glm::reflect(glm::normalize(r.direction()), rec.normal);
                throughput = throughputPrev;
                // 将反射率按选择概率归一化，确保无偏（BRDF/p(selection)).
                if (pSpec > 0.0f) throughput *= rec.albedo / pSpec;
                r = Ray(rec.point + rec.normal * 0.001f, glm::normalize(refl));
                currBsdfPdf = 0.0f;
                currBounceDiffuse = false;
            } else {
                // 漫反射分支：根据开关选择是否使用余弦加权半球重要性采样
                if (pDiff <= 0.0f) {
                    break;
                }

                bool allowDirect = (lightingMode != LIGHTING_MODE_INDIRECT);
                bool allowIndirect = (lightingMode != LIGHTING_MODE_DIRECT);

                float invPDiff = 1.0f / pDiff;
                glm::vec3 baseWeight = throughputPrev * invPDiff;
                glm::vec3 brdf = rec.albedo / kPi;

                glm::vec3 bounceDir(0.0f);
                float cosThetaBounce = 0.0f;
                float pdfBsdfSample = 0.0f;
                if (allowIndirect || lightingMode == LIGHTING_MODE_MIS) { // 间接光
                    if (enableDiffuseImportanceSampling) {
                        bounceDir = cosineSampleHemisphere(rec.normal, seed);
                        cosThetaBounce = fmaxf(glm::dot(rec.normal, bounceDir), 0.0f);
                    } else {
                        bounceDir = uniformSampleHemisphere(rec.normal, seed);
                        cosThetaBounce = fmaxf(glm::dot(rec.normal, bounceDir), 0.0f);
                    }
                    pdfBsdfSample = bsdfPdf(enableDiffuseImportanceSampling, cosThetaBounce);
                }

                if (allowDirect) {
                    radiance += sampleDirectLightingContribution(models, modelCount, illuminants, illuminantCount, seed, rec.point, rec.normal, brdf, baseWeight, pDiff, enableDiffuseImportanceSampling, lightingMode);
                }

                if (!allowIndirect) {
                    prevBsdfPdf = 0.0f;
                    prevBounceDiffuse = false;
                    break;
                }

                throughput = throughputPrev * rec.albedo * invPDiff;
                r = Ray(rec.point + rec.normal * 0.001f, bounceDir);
                currBounceDiffuse = true;
                currBsdfPdf = pDiff > 0.0f ? pDiff * pdfBsdfSample : 0.0f;
            }
            // 俄罗斯轮盘赌：当 throughput 很小时随机终止以减少无用的追踪路径。
            // 使用亮度作为衡量（L1或L2均可），并保证在继续时对 throughput 进行重要性权重校正以保持无偏。
            if (enableRussianRoulette) {
                int startDepth = glm::max(1, rouletteStartDepth);
                if (depth >= startDepth) {
                    float q = fmaxf(0.05f, 1.0f - glm::max(throughput.r, glm::max(throughput.g, throughput.b))); // 最小保留概率
                    float contProb = 1.0f - q;
                    float rn = rand01(seed);
                    if (rn < q || contProb <= 0.0f) {
                        break;
                    }
                    throughput /= contProb;
                }
            }

            prevBsdfPdf = currBsdfPdf;
            prevBounceDiffuse = currBounceDiffuse;
        }
        return radiance;
    }

    // CUDA核函数：每像素采样多次，累积结果
    __global__ void pathTraceKernel(
        const ModelGPU* models, int modelCount,
        const IlluminantGPU* illuminants, int illuminantCount,
        unsigned int *pbo, int width, int height, int samplesPerPixel, int maxDepth,
        LightingMode lightingMode, bool enableDiffuseImportanceSampling,
        bool enableRussianRoulette, int rouletteStartDepth,
        glm::vec3 camPos, glm::vec3 lowerLeft, glm::vec3 horizontal, glm::vec3 vertical,
        glm::vec3 *accumBuffer, int accumFrameCount, int frameIndex) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;  

        uint32_t seed = (frameIndex + 1) * 9781u + x * 6271u + y * 1259u; 
        glm::vec3 color(0.0f); // 累积当前像素所有样本的线性颜色（未平均前）
        for (int s = 0; s < samplesPerPixel; ++s) { // 多次采样以降低蒙特卡洛噪声
            float u = (x + rand01(seed)) / static_cast<float>(width - 1);  // 水平归一化坐标 + 随机抖动（亚像素采样，抗锯齿）
            float v = (y + rand01(seed)) / static_cast<float>(height - 1); // 垂直归一化坐标 + 随机抖动
            glm::vec3 dir = lowerLeft + u * horizontal + v * vertical - camPos; // 计算该随机子像素对应的射线方向（相机透视投射）
            Ray r(camPos, glm::normalize(dir)); // 构造从相机位置出发的单位方向射线

            color += traceRay(models, modelCount,
                              illuminants, illuminantCount,
                              r, seed, maxDepth,
                              lightingMode, enableDiffuseImportanceSampling,
                              enableRussianRoulette, rouletteStartDepth); // 路径追踪
        }
        color /= static_cast<float>(samplesPerPixel); // 将累积的总和除以采样数目得到平均颜色
        color = glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f)); // 保证数值在显示合法范围内，避免溢出
        int pixelIdx = y * width + x; // 二维像素坐标映射到线性索引

        // 支持累积多帧（降噪）
        if (accumBuffer) {
            glm::vec3 accumValue = accumBuffer[pixelIdx];
            accumValue += color;
            accumBuffer[pixelIdx] = accumValue;
            color = accumValue / static_cast<float>(accumFrameCount);
        }

        // Gamma校正（sqrt）并写入PBO
        color = glm::sqrt(color);
        unsigned int r8 = static_cast<unsigned int>(255.99f * color.r);
        unsigned int g8 = static_cast<unsigned int>(255.99f * color.g);
        unsigned int b8 = static_cast<unsigned int>(255.99f * color.b);
        pbo[pixelIdx] = (0xFFu << 24) | (b8 << 16) | (g8 << 8) | r8;
    }
} // namespace

extern "C" void launchPathTracer(
    unsigned int *pbo, int width, int height, const Camera &camera, 
    const ModelGPU* models, int modelCount,
    const IlluminantGPU* illuminants, int illuminantCount,
    int samplesPerPixel, int maxDepth,
    LightingMode lightingMode, bool enableDiffuseImportanceSampling,
    bool enableRussianRoulette, int rouletteStartDepth,
    glm::vec3 *accumBuffer, int accumFrameCount, int frameIndex) 
{
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    float fovRad = camera.getFov() * 0.0174532925f;
    float viewportHeight = 2.0f * tanf(fovRad * 0.5f);
    float viewportWidth = aspect * viewportHeight;
    glm::vec3 camPos = camera.getPosition();
    glm::vec3 front = glm::normalize(camera.getFront());
    glm::vec3 right = glm::normalize(camera.getRight());
    glm::vec3 up = glm::normalize(camera.getUp());
    glm::vec3 horizontal = viewportWidth * right;
    glm::vec3 vertical = viewportHeight * up;
    glm::vec3 lowerLeft = camPos + front - horizontal * 0.5f - vertical * 0.5f;
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    pathTraceKernel<<<grid, block>>>(models, modelCount,
                                     illuminants, illuminantCount,
                                     pbo, width, height, samplesPerPixel, maxDepth,
                                     lightingMode, enableDiffuseImportanceSampling,
                                     enableRussianRoulette, rouletteStartDepth,
                                     camPos, lowerLeft, horizontal, vertical,
                                     accumBuffer, accumFrameCount, frameIndex);
}
