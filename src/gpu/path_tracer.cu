#include "path_tracer.h"
#include "scene/camera.h"
#include "scene/shape.h"
#include "scene/Ray.h"
#include "gpu/pt_utils.h"

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
    // 遍历 Shape 数组求交，返回最近命中
    __device__ bool hitShapes(const Shape* shapes, int shapeCount, const Ray& r, float tMin, float tMax, HitRecord& rec) {
        bool hitAnything = false;
        float closest = tMax;
        for (int i = 0; i < shapeCount; ++i) {
            const Shape& s = shapes[i];
            HitRecord temp;
            bool hit = false;
            switch (s.type) {
                case SHAPE_SPHERE: {
                    // 球体相交
                    glm::vec3 oc = r.origin() - s.data.sph.center;
                    float a = glm::dot(r.direction(), r.direction());
                    float halfB = glm::dot(oc, r.direction());
                    float c = glm::dot(oc, oc) - s.data.sph.radius * s.data.sph.radius;
                    float discriminant = halfB * halfB - a * c;
                    if (discriminant >= 0.0f) {
                        float sqrtD = sqrtf(discriminant);
                        float root = (-halfB - sqrtD) / a;
                        if (root < tMin || root > closest) {
                            root = (-halfB + sqrtD) / a;
                        }
                        if (root >= tMin && root <= closest) {
                            temp.t = root;
                            temp.point = r.at(root);
                            temp.normal = glm::normalize((temp.point - s.data.sph.center) / s.data.sph.radius);
                            temp.albedo = s.data.sph.mat.albedo;
                            temp.metallic = s.data.sph.mat.metallic;
                            temp.emission = s.data.sph.mat.emission;
                            hit = true;
                        }
                    }
                } break;
                case SHAPE_TRIANGLE: {
                    // Möller-Trumbore
                    const glm::vec3& v0 = s.data.tri.v0;
                    const glm::vec3& v1 = s.data.tri.v1;
                    const glm::vec3& v2 = s.data.tri.v2;
                    glm::vec3 edge1 = v1 - v0;
                    glm::vec3 edge2 = v2 - v0;
                    glm::vec3 pvec = glm::cross(r.direction(), edge2);
                    float det = glm::dot(edge1, pvec);
                    if (fabsf(det) > 1e-6f) {
                        float invDet = 1.0f / det;
                        glm::vec3 tvec = r.origin() - v0;
                        float u = glm::dot(tvec, pvec) * invDet;
                        if (u >= 0.0f && u <= 1.0f) {
                            glm::vec3 qvec = glm::cross(tvec, edge1);
                            float v = glm::dot(r.direction(), qvec) * invDet;
                            if (v >= 0.0f && u + v <= 1.0f) {
                                float t = glm::dot(edge2, qvec) * invDet;
                                if (t >= tMin && t <= closest) {
                                    temp.t = t;
                                    temp.point = r.at(t);
                                    temp.normal = glm::normalize(glm::cross(edge1, edge2));
                                    temp.albedo = s.data.tri.mat.albedo;
                                    temp.metallic = s.data.tri.mat.metallic;
                                    temp.emission = s.data.tri.mat.emission;
                                    hit = true;
                                }
                            }
                        }
                    }
                } break;
                case SHAPE_PLANE: {
                    const glm::vec3& n = s.data.pln.normal;
                    float denom = glm::dot(n, r.direction());
                    if (fabsf(denom) > 1e-6f) {
                        float t = glm::dot(s.data.pln.point - r.origin(), n) / denom;
                        if (t >= tMin && t <= closest) {
                            temp.t = t;
                            temp.point = r.at(t);
                            temp.normal = n;
                            temp.albedo = s.data.pln.mat.albedo;
                            temp.metallic = s.data.pln.mat.metallic;
                            temp.emission = s.data.pln.mat.emission;
                            hit = true;
                        }
                    }
                } break;
            }
            if (hit) {
                hitAnything = true;
                closest = temp.t;
                rec = temp;
            }
        }
        return hitAnything;
    }

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
    __device__ glm::vec3 traceRay(const Shape* shapes, int shapeCount, Ray r, uint32_t &seed, int maxDepth) {
        glm::vec3 radiance(0.0f);
        glm::vec3 throughput(1.0f);
        for (int depth = 0; depth < maxDepth; ++depth) {
            HitRecord rec;
            // 场景求交，未命中返回环境光
            if (!hitShapes(shapes, shapeCount, r, 0.001f, FLT_MAX, rec)) {
                radiance += throughput * environmentColor(r);
                break;
            }
            // 自发光累加
            radiance += throughput * rec.emission;

            // Fresnel概率混合采样
            float metal = rec.metallic;
            glm::vec3 inDir = glm::normalize(r.direction());
            float cosTheta = fmaxf(glm::dot(-inDir, rec.normal), 0.0f);
            glm::vec3 F0 = glm::mix(glm::vec3(0.04f), rec.albedo, metal);
            glm::vec3 F = fresnelSchlick_fn(cosTheta, F0);
            float specProb = (F.r + F.g + F.b) / 3.0f;
            specProb = fminf(fmaxf(specProb, 0.01f), 0.99f); // 保证概率不为0或1
            float rchoice = rand01(seed);

            if (rchoice < specProb) {
                // 镜面反射分支
                glm::vec3 refl = glm::reflect(inDir, rec.normal);
                throughput *= F;
                throughput /= specProb;
                r = Ray(rec.point + rec.normal * 0.001f, glm::normalize(refl));
            } else {
                // 漫反射分支
                glm::vec3 bounceDir = cosineSampleHemisphere(rec.normal, seed);
                throughput *= rec.albedo;
                throughput /= (1.0f - specProb);
                r = Ray(rec.point + rec.normal * 0.001f, bounceDir);
            }
        }
        return radiance;
    }

    // CUDA核函数：每像素采样多次，累积结果
    __global__ void pathTraceKernel(
        const Shape* shapes, int shapeCount,
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
            
            color += traceRay(shapes, shapeCount, r, seed, maxDepth); // 路径追踪
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

extern "C" void launchPathTracer(unsigned int *pbo, int width, int height, const Camera &camera, const Shape* shapes, int shapeCount, int samplesPerPixel, int maxDepth, glm::vec3 *accumBuffer, int accumFrameCount, int frameIndex) {
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
    pathTraceKernel<<<grid, block>>>(shapes, shapeCount, pbo, width, height, samplesPerPixel, maxDepth, camPos, lowerLeft, horizontal, vertical, accumBuffer, accumFrameCount, frameIndex);
    cudaDeviceSynchronize();
}
