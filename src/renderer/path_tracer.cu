#include "renderer/path_tracer.h"
#include "renderer/intersect.h"
#include "renderer/pt_utils.h"
#include "struct/ray.h"
#include "struct/shapeGpu.h"
#include "struct/bvhNodeGpu.h"
#include "struct/materialGpu.h"
#include <curand_kernel.h>

// 前向声明
__device__ glm::vec3 traceRay(
    Ray r,
    curandState* seed,
    const TriangleGPU* triangles, int triangleCount,
    const MaterialGpu* materials,
    const BVHNodeGPU* bvhNodes,
    const int* triIndices,
    const int* lightIndices, int lightCount,
    int maxDepth, bool enableRussianRoulette, int rouletteStartDepth,
    bool enableDiffuseImportanceSampling, LightingMode lightingMode
);

// 主内核
extern "C" __global__ void kernel_path_tracer_impl(
    uint8_t* pbo_ptr, int width, int height,
    glm::vec3 cam_pos, glm::mat4 cam_view, float cam_fov,
    glm::vec3* accumulated_radiance, uint32_t frame_count,
    const TriangleGPU* triangles, int triangle_count,
    const MaterialGpu* materials,
    const BVHNodeGPU* bvh_nodes,
    const int* tri_indices,
    const int* light_indices, int light_count,
    int max_depth, int samples_per_pixel, bool enable_rr, int rr_depth,
    bool enable_diffuse_is, LightingMode lighting_mode
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int pixel_index = y * width + x;
    curandState seed;
    curand_init(width * height * frame_count + pixel_index, 0, 0, &seed);

    glm::vec3 total_radiance(0.0f);
    for (int s = 0; s < samples_per_pixel; ++s) {
        float u = (float(x) + curand_uniform(&seed)) / float(width);
        float v = (float(y) + curand_uniform(&seed)) / float(height);
        Ray r = generateCameraRay(u, v, cam_pos, cam_view, cam_fov, width, height);
        total_radiance += traceRay(r, &seed, triangles, triangle_count, materials, bvh_nodes, tri_indices, light_indices, light_count, max_depth, enable_rr, rr_depth, enable_diffuse_is, lighting_mode);
    }

    glm::vec3 current_radiance = total_radiance / float(samples_per_pixel);
    glm::vec3 previous_radiance = accumulated_radiance[pixel_index];
    float mix_factor = 1.0f / float(frame_count + 1);
    glm::vec3 final_radiance = glm::mix(previous_radiance, current_radiance, mix_factor);
    accumulated_radiance[pixel_index] = final_radiance;

    writePixel(pbo_ptr, width, x, y, final_radiance);
}

// 包装函数以满足头文件
void kernel_path_tracer(
    uint8_t* pbo_ptr, int width, int height,
    const glm::vec3& cam_pos, const glm::mat4& cam_view, float cam_fov,
    glm::vec3* accumulated_radiance, uint32_t frame_count,
    const TriangleGPU* triangles, int triangle_count,
    const MaterialGpu* materials,
    const BVHNodeGPU* bvh_nodes,
    const int* tri_indices,
    const int* light_indices, int light_count,
    int max_depth, int samples_per_pixel, bool enable_rr, int rr_depth,
    bool enable_diffuse_is, LightingMode lighting_mode
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernel_path_tracer_impl<<<grid, block>>>(
        pbo_ptr, width, height,
        cam_pos, cam_view, cam_fov,
        accumulated_radiance, frame_count,
        triangles, triangle_count, materials, bvh_nodes, tri_indices,
        light_indices, light_count,
        max_depth, samples_per_pixel, enable_rr, rr_depth,
        enable_diffuse_is, lighting_mode
    );
}

__device__ float lightPdfFromHit(const TriangleGPU* triangles, const int* lightIndices, int lightCount, int lightTriIndex, const glm::vec3& origin, const glm::vec3& hit_p, const glm::vec3& hit_n) {
    if (lightCount == 0) return 0.0f;
    const TriangleGPU& tri = triangles[lightIndices[lightTriIndex]];
    float area = 0.5f * glm::length(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
    if (area < 1e-6f) return 0.0f;
    
    glm::vec3 to_light = hit_p - origin;
    float dist_sq = glm::dot(to_light, to_light);
    if (dist_sq < 1e-6f) return 0.0f;
    
    float cos_theta_light = fabsf(glm::dot(glm::normalize(to_light), hit_n));
    return dist_sq / (cos_theta_light * area * lightCount);
}

__device__ glm::vec3 sampleDirectLightingContribution(
    const TriangleGPU* triangles, int triangleCount,
    const int* lightIndices, int lightCount,
    curandState* seed,
    const glm::vec3& p, const glm::vec3& n, const glm::vec3& brdf,
    const glm::vec3& throughput, float baseWeight,
    bool enableDiffuseIS, LightingMode lightingMode,
    const BVHNodeGPU* bvhNodes, const int* triIndices, const MaterialGpu* materials)
{
    if (lightCount == 0) return glm::vec3(0.0f);

    // 采样光源
    int light_idx = int(curand_uniform(seed) * lightCount);
    const TriangleGPU& light_tri = triangles[lightIndices[light_idx]];

    // 在光源上采样一个点
    float r1 = curand_uniform(seed);
    float r2 = curand_uniform(seed);
    if (r1 + r2 > 1.0f) { r1 = 1.0f - r1; r2 = 1.0f - r2; }
    glm::vec3 light_p = light_tri.v0 * (1.0f - r1 - r2) + light_tri.v1 * r1 + light_tri.v2 * r2;
    glm::vec3 light_n = glm::normalize(glm::cross(light_tri.v1 - light_tri.v0, light_tri.v2 - light_tri.v0));

    glm::vec3 to_light = light_p - p;
    float dist_to_light_sq = glm::dot(to_light, to_light);
    glm::vec3 dir_to_light = glm::normalize(to_light);

    // 检查可见性
    Ray shadow_ray(p + n * 0.001f, dir_to_light);
    HitRecord shadow_rec;
    bool in_shadow = shadowIntersectScene(shadow_ray, 0.001f, sqrtf(dist_to_light_sq) - 0.002f, bvhNodes, triangles, triIndices);

    if (in_shadow) {
        return glm::vec3(0.0f);
    }

    // 计算 PDF 和贡献
    float light_area = 0.5f * glm::length(glm::cross(light_tri.v1 - light_tri.v0, light_tri.v2 - light_tri.v0));
    float pdf_light = dist_to_light_sq / (fabsf(glm::dot(light_n, -dir_to_light)) * light_area * lightCount);
    
    float cos_theta_surf = fmaxf(0.0f, glm::dot(n, dir_to_light));
    if (cos_theta_surf < 1e-6f || pdf_light < 1e-6f) {
        return glm::vec3(0.0f);
    }

    const MaterialGpu& light_mat = materials[light_tri.materialIndex];
    glm::vec3 emission = light_mat.emission;

    if (lightingMode == LIGHTING_MODE_MIS) {
        float pdf_bsdf = bsdfPdf(enableDiffuseIS, cos_theta_surf);
        float weight = powerHeuristic(pdf_light, pdf_bsdf);
        return (brdf * emission * cos_theta_surf / pdf_light) * weight * baseWeight;
    } else { // 直接或间接
        return (brdf * emission * cos_theta_surf / pdf_light) * baseWeight;
    }
}


__device__ glm::vec3 traceRay(
    Ray r,
    curandState* seed,
    const TriangleGPU* triangles, int triangleCount,
    const MaterialGpu* materials,
    const BVHNodeGPU* bvhNodes,
    const int* triIndices,
    const int* lightIndices, int lightCount,
    int maxDepth, bool enableRussianRoulette, int rouletteStartDepth,
    bool enableDiffuseImportanceSampling, LightingMode lightingMode
) {
    glm::vec3 radiance(0.0f);
    glm::vec3 throughput(1.0f);
    bool prevBounceDiffuse = false;
    float prevBsdfPdf = 0.0f;

    for (int depth = 0; depth < maxDepth; ++depth) {
        if (glm::dot(throughput, throughput) < 1e-9f) {
            continue;
        }

        HitRecord rec;
        bool hit = intersectScene(r, 0.001f, FLT_MAX, rec, bvhNodes, triangles, triIndices, materials);

        if (!hit) {
            radiance += throughput * environmentColor(r);
            break;
        }
        
        const MaterialGpu& mat = materials[rec.materialIndex];

        if (glm::dot(mat.emission, mat.emission) > 0.0f) {
            float weight = 1.0f;
            if (lightingMode == LIGHTING_MODE_MIS && prevBounceDiffuse && prevBsdfPdf > 0.0f) {
                // 找到击中的光源三角形
                int lightTriIndex = -1;
                for(int i=0; i<lightCount; ++i) {
                    if(lightIndices[i] == rec.primitiveIndex) { // primitiveIndex 应该是全局三角形索引
                        lightTriIndex = i;
                        break;
                    }
                }
                if (lightTriIndex != -1) {
                    float pdfLight = lightPdfFromHit(triangles, lightIndices, lightCount, lightTriIndex, r.origin(), rec.point, rec.normal);
                    if (pdfLight > 0.0f) {
                        weight = powerHeuristic(prevBsdfPdf, pdfLight);
                    }
                }
            }
            radiance += throughput * mat.emission * weight;
            break; 
        }

        float currBsdfPdf = 0.0f;
        bool currBounceDiffuse = false;
        float metallic = mat.metallic;

        // 计算基础反射率 F0
        glm::vec3 F0 = glm::mix(glm::vec3(0.04f), mat.albedo, metallic);

        glm::vec3 directLighting(0.0f);
        if (lightingMode != LIGHTING_MODE_INDIRECT) {
            // 漫反射部分
            if (metallic < 1.0f) {
                glm::vec3 diffuseBrdf = mat.albedo / kPi * (1.0f - metallic);
                directLighting += sampleDirectLightingContribution(
                    triangles, triangleCount, lightIndices, lightCount, seed,
                    rec.point, rec.normal, diffuseBrdf, throughput, 1.0f,
                    enableDiffuseImportanceSampling, lightingMode,
                    bvhNodes, triIndices, materials
                );
            }
            // 镜面反射部分：简化，仅在直接光照时近似
            if (metallic > 0.0f) {
                glm::vec3 specularBrdf = F0;
                directLighting += sampleDirectLightingContribution(
                    triangles, triangleCount, lightIndices, lightCount, seed,
                    rec.point, rec.normal, specularBrdf, throughput, metallic,
                    false, lightingMode,
                    bvhNodes, triIndices, materials
                );
            }
        }
        radiance += directLighting;

        if (lightingMode != LIGHTING_MODE_DIRECT) {
            // 根据 metallic 混合反射方向
            glm::vec3 bounceDir;
            glm::vec3 nextBrdf;
            float pdf;

            if (curand_uniform(seed) < metallic) {
                // 镜面反射采样（使用 GGX）
                float roughness = getRoughnessFromMetallic(metallic);
                glm::vec3 H;
                float pdfH;
                H = sampleGGX(rec.normal, roughness, seed, H, pdfH);
                bounceDir = glm::reflect(r.direction(), H);
                
                // 确保在表面之上
                if (glm::dot(bounceDir, rec.normal) < 1e-6f) {
                    bounceDir = glm::reflect(r.direction(), rec.normal);
                }
                
                float NoV = fmaxf(0.0f, glm::dot(rec.normal, -r.direction()));
                float NoL = fmaxf(0.0f, glm::dot(rec.normal, bounceDir));
                float NoH = fmaxf(0.0f, glm::dot(rec.normal, H));
                float VoH = fmaxf(0.0f, glm::dot(-r.direction(), H));

                if (NoL > 1e-6f && NoV > 1e-6f && NoH > 1e-6f && VoH > 1e-6f) {
                    float D = ggxDistribution(NoH, roughness);
                    float G = smithGeometry(NoV, NoL, roughness);
                    glm::vec3 F = fresnelSchlick(VoH, F0);
                    glm::vec3 spec = (D * G * F) / (4.0f * NoV * NoL);
                    nextBrdf = spec;
                    pdf = pdfH / (4.0f * VoH + 1e-6f);
                } else {
                    nextBrdf = glm::vec3(0.0f);
                    pdf = 1e-6f;
                }
                currBounceDiffuse = false;
            } else {
                // 漫反射采样
                bounceDir = enableDiffuseImportanceSampling ? 
                    cosineSampleHemisphere(rec.normal, seed) : 
                    uniformSampleHemisphere(rec.normal, seed);
                
                float cosTheta = fmaxf(0.0f, glm::dot(rec.normal, bounceDir));
                pdf = bsdfPdf(enableDiffuseImportanceSampling, cosTheta);
                
                glm::vec3 diffuse = mat.albedo / kPi * (1.0f - metallic);
                nextBrdf = diffuse * cosTheta;
                currBounceDiffuse = true;
            }

            // 更新 throughput
            if (pdf > 1e-6f) {
                throughput *= nextBrdf / pdf;
            } else {
                throughput = glm::vec3(0.0f);
            }

            r = Ray(rec.point + rec.normal * 0.001f, bounceDir);
            prevBsdfPdf = pdf;
        } else {
            break; // 直接光照模式下不再反弹
        }


        if (enableRussianRoulette && depth >= rouletteStartDepth) {
            float max_comp = fmaxf(throughput.r, fmaxf(throughput.g, throughput.b));
            if (curand_uniform(seed) > max_comp) {
                break;
            }
            throughput /= max_comp;
        }
    }
    return radiance;
}
