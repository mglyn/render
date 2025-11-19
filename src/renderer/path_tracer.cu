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

    // Fresnel-Schlick近似，用于镜面概率混合
    __device__ glm::vec3 fresnelSchlick_fn(float cosTheta, const glm::vec3 &F0) {
        return F0 + (glm::vec3(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
    }

    // 路径追踪主循环：Fresnel概率混合采样，累积throughput
    __device__ glm::vec3 traceRay(const Shape* shapes, int shapeCount, const ModelGPU* models, int modelCount, Ray r, uint32_t &seed, int maxDepth) {
        glm::vec3 radiance(0.0f);
        glm::vec3 throughput(1.0f);
        for (int depth = 0; depth < maxDepth; ++depth) {
            // 统一求交：同时检查BVH和基础形状，选择最近命中
            HitRecord rec;
            bool hit = false;
            float closest = FLT_MAX;
            
            // 检查BVH模型
            if (modelCount > 0 && models) {
                HitRecord bvhRec;
                if (intersectBVH(models, modelCount, r, 0.001f, closest, bvhRec)) {
                    hit = true;
                    closest = bvhRec.t;
                    rec = bvhRec;
                }
            }
            
            // 检查基础形状（球体、平面等）
            if (shapeCount > 0 && shapes) {
                HitRecord shapeRec;
                if (intersect(shapes, shapeCount, r, 0.001f, closest, shapeRec)) {
                    hit = true;
                    closest = shapeRec.t;
                    rec = shapeRec;
                }
            }
            
            // 未命中返回环境光
            if (!hit) {
                radiance += throughput * environmentColor(r);
                break;
            }

            // 自发光累加
            radiance += throughput * rec.emission;

            // 基于金属度的简单混合（离散混合采样）。
            // 为了保持无偏，需要对选择的采样策略按其选择概率进行重要性权重校正。
            float pSpec = rec.metallic;
            float pDiff = 1.0f - pSpec;
            float choice = rand01(seed);
            if (choice < pSpec) {
                // 镜面反射分支（delta分布）。
                glm::vec3 refl = glm::reflect(glm::normalize(r.direction()), rec.normal);
                // 将反射率按选择概率归一化，确保无偏（BRDF/p(selection)).
                if (pSpec > 0.0f) throughput *= rec.albedo / pSpec;
                r = Ray(rec.point + rec.normal * 0.001f, glm::normalize(refl));
            } else {
                // 漫反射分支（余弦加权半球采样）。
                glm::vec3 bounceDir = cosineSampleHemisphere(rec.normal, seed);
                if (pDiff > 0.0f) throughput *= rec.albedo / pDiff;
                r = Ray(rec.point + rec.normal * 0.001f, bounceDir);

                // === 直接光照：向光源采样 + pdf修正 ===
                // 只采样球体光源
                int lightCount = 0;
                for (int i = 0; i < shapeCount; ++i) {
                    if (shapes[i].type == SHAPE_SPHERE &&
                        (shapes[i].data.sph.mat.emission.r > 0.0f || shapes[i].data.sph.mat.emission.g > 0.0f || shapes[i].data.sph.mat.emission.b > 0.0f)) {
                        ++lightCount;
                    }
                }
                if (lightCount > 0) {
                    int lightIdx = static_cast<int>(rand01(seed) * lightCount);
                    int found = 0;
                    const Shape* light = nullptr;
                    for (int i = 0; i < shapeCount; ++i) {
                        if (shapes[i].type == SHAPE_SPHERE &&
                            (shapes[i].data.sph.mat.emission.r > 0.0f || shapes[i].data.sph.mat.emission.g > 0.0f || shapes[i].data.sph.mat.emission.b > 0.0f)) {
                            if (found == lightIdx) {
                                light = &shapes[i];
                                break;
                            }
                            ++found;
                        }
                    }
                    if (light) {
                        // 球面均匀采样
                        float u1 = rand01(seed);
                        float u2 = rand01(seed);
                        float z = 1.0f - 2.0f * u1;
                        float phi = 2.0f * 3.1415926535f * u2;
                        float rxy = sqrtf(1.0f - z * z);
                        glm::vec3 onLight = light->data.sph.center + light->data.sph.radius * glm::vec3(rxy * cosf(phi), rxy * sinf(phi), z);
                        glm::vec3 toLight = onLight - rec.point;
                        float dist2 = glm::dot(toLight, toLight);
                        float dist = sqrtf(dist2);
                        glm::vec3 toLightDir = toLight / dist;
                        // 检查可见性（shadow ray）
                        Ray shadowRay(rec.point + rec.normal * 0.001f, toLightDir);
                        bool occluded = false;
                        for (int i = 0; i < shapeCount; ++i) {
                            if (&shapes[i] == light) continue;
                            HitRecord tmpRec;
                            if (intersect(&shapes[i], 1, shadowRay, 0.001f, dist - 0.001f, tmpRec)) {
                                occluded = true;
                                break;
                            }
                        }
                        if (!occluded) {
                            // pdf = 1 / (4 * pi * r^2) * 1 / lightCount
                            float lightArea = 4.0f * 3.1415926535f * light->data.sph.radius * light->data.sph.radius;
                            float pdf = (1.0f / lightArea) * (1.0f / lightCount);
                            float cosTheta = glm::dot(-toLightDir, rec.normal);
                            float cosThetaL = glm::dot(toLightDir, (onLight - light->data.sph.center) / light->data.sph.radius);
                            if (cosTheta > 0.0f && cosThetaL > 0.0f) {
                                // 距离平方衰减和法线夹角修正
                                float G = cosTheta * cosThetaL / dist2;
                                glm::vec3 Le = light->data.sph.mat.emission;
                                // Lambert BRDF = albedo / pi
                                glm::vec3 brdf = rec.albedo / 3.1415926535f;
                                radiance += throughput * Le * brdf * G / pdf;
                            }
                        }
                    }
                }
            }
            // 俄罗斯轮盘赌：当 throughput 很小时随机终止以减少无用的追踪路径。
            // 使用亮度作为衡量（L1或L2均可），并保证在继续时对 throughput 进行重要性权重校正以保持无偏。
            float q = fmaxf(0.05f, 1.0f - glm::max(throughput.r, glm::max(throughput.g, throughput.b))); // 最小保留概率
            // q 是“终止概率”，实际继续概率为 (1 - q)。若 (1 - q) 很小则直接终止以避免数值问题。
            float contProb = 1.0f - q;
            if (depth > 2) { // 给几个深度以保留初始重要路径
                float rn = rand01(seed);
                if (rn < q || contProb <= 0.0f) {
                    break; // 终止路径
                }
                // 如果继续，需要将 throughput 除以继续概率以纠正期望（无偏性）
                throughput /= contProb;
            }
        }
        return radiance;
    }

    // CUDA核函数：每像素采样多次，累积结果
    __global__ void pathTraceKernel(
        const Shape* shapes, int shapeCount,
        const ModelGPU* models, int modelCount,
        unsigned int *pbo, int width, int height, int samplesPerPixel, int maxDepth,
        glm::vec3 camPos, glm::vec3 lowerLeft, glm::vec3 horizontal, glm::vec3 vertical,
        glm::vec3 *accumBuffer, int accumFrameCount, int frameIndex) {
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
            
            color += traceRay(shapes, shapeCount, models, modelCount, r, seed, maxDepth); // 路径追踪
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

extern "C" void launchPathTracer(unsigned int *pbo, int width, int height, const Camera &camera, 
    const Shape* shapes, int shapeCount, 
    const ModelGPU* models, int modelCount,
    int samplesPerPixel, int maxDepth, glm::vec3 *accumBuffer, int accumFrameCount, int frameIndex) {
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
    pathTraceKernel<<<grid, block>>>(shapes, shapeCount, models, modelCount, pbo, width, height, samplesPerPixel, maxDepth, camPos, lowerLeft, horizontal, vertical, accumBuffer, accumFrameCount, frameIndex);
    cudaDeviceSynchronize();
}
