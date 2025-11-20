#include "renderer/pt_utils.h"
#include "renderer/hitRecord.h"

__device__ void writePixel(uint8_t* pbo_ptr, int width, int x, int y, const glm::vec3& color) {
    int r = static_cast<int>(glm::clamp(color.r, 0.0f, 1.0f) * 255.0f);
    int g = static_cast<int>(glm::clamp(color.g, 0.0f, 1.0f) * 255.0f);
    int b = static_cast<int>(glm::clamp(color.b, 0.0f, 1.0f) * 255.0f);
    int a = 255;
    int pixel_index = (y * width + x) * 4;
    pbo_ptr[pixel_index + 0] = r;
    pbo_ptr[pixel_index + 1] = g;
    pbo_ptr[pixel_index + 2] = b;
    pbo_ptr[pixel_index + 3] = a;
}

__device__ Ray generateCameraRay(float u, float v, const glm::vec3& cam_pos, const glm::mat4& cam_view, float fov, int width, int height) {
    float aspect = static_cast<float>(width) / height;
    float fov_rad = fov * kPi / 180.0f;
    float viewport_height = 2.0f * tanf(fov_rad * 0.5f);
    float viewport_width = aspect * viewport_height;

    glm::vec3 origin = cam_pos;
    glm::vec3 horizontal = glm::vec3(cam_view[0][0], cam_view[1][0], cam_view[2][0]) * viewport_width;
    glm::vec3 vertical = glm::vec3(cam_view[0][1], cam_view[1][1], cam_view[2][1]) * viewport_height;
    glm::vec3 lower_left_corner = origin - horizontal * 0.5f - vertical * 0.5f - glm::vec3(cam_view[0][2], cam_view[1][2], cam_view[2][2]);

    return Ray(origin, glm::normalize(lower_left_corner + u * horizontal + v * vertical - origin));
}

__device__ glm::vec3 environmentColor(const Ray& r) {
    glm::vec3 unit_direction = glm::normalize(r.direction());
    float t = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - t) * glm::vec3(0.05f, 0.04f, 0.05f) + t * glm::vec3(0.1f, 0.05f, 0.1f);
}

__device__ glm::vec3 uniformSampleHemisphere(const glm::vec3& normal, curandState* seed) {
    float u1 = curand_uniform(seed);
    float u2 = curand_uniform(seed);
    float r = sqrtf(1.0f - u1 * u1);
    float phi = 2.0f * kPi * u2;
    glm::vec3 sample(r * cosf(phi), r * sinf(phi), u1);

    glm::vec3 w = normal;
    glm::vec3 u = glm::normalize(glm::cross((fabs(w.x) > 0.1f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w));
    glm::vec3 v = glm::cross(w, u);

    return u * sample.x + v * sample.y + w * sample.z;
}

__device__ glm::vec3 cosineSampleHemisphere(const glm::vec3& normal, curandState* seed) {
    float u1 = curand_uniform(seed);
    float u2 = curand_uniform(seed);
    float r = sqrtf(u1);
    float theta = 2.0f * kPi * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    glm::vec3 w = normal;
    glm::vec3 u = glm::normalize(glm::cross((fabs(w.x) > 0.1f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w));
    glm::vec3 v = glm::cross(w, u);

    return glm::normalize(u * x + v * y + w * z);
}

__device__ float bsdfPdf(bool useCosineSampling, float cosTheta) {
    return useCosineSampling ? fmaxf(0.f, cosTheta / kPi) : 1.0f / (2.0f * kPi);
}

__device__ float powerHeuristic(float pdfA, float pdfB) {
    float a = pdfA;
    float b = pdfB;
    return (a * a) / (a * a + b * b);
}

// PBR 镜面反射函数
__device__ glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0) {
    return F0 + (glm::vec3(1.0f) - F0) * powf(fmaxf(0.0f, 1.0f - cosTheta), 5.0f);
}

// GGX 法线分布函数
__device__ float ggxDistribution(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * denom * denom + 1e-6f);
}

// Smith G1 函数
__device__ float smithG1(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k + 1e-6f);
}

// Smith 几何遮挡函数
__device__ float smithGeometry(float NdotV, float NdotL, float roughness) {
    return smithG1(NdotV, roughness) * smithG1(NdotL, roughness);
}

// GGX 半程向量采样
__device__ glm::vec3 sampleGGX(const glm::vec3& n, float roughness, curandState* seed, glm::vec3& H, float& pdf) {
    float u1 = curand_uniform(seed);
    float u2 = curand_uniform(seed);

    float a = roughness * roughness;
    float phi = 2.0f * kPi * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (a * a - 1.0f) * u2));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    // 本地坐标
    glm::vec3 H_local(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

    // 构建正交基
    glm::vec3 up = fabsf(n.z) < 0.999f ? glm::vec3(0,0,1) : glm::vec3(1,0,0);
    glm::vec3 tangent = glm::normalize(glm::cross(up, n));
    glm::vec3 bitangent = glm::cross(n, tangent);
    H = glm::normalize(tangent * H_local.x + bitangent * H_local.y + n * H_local.z);

    float D = ggxDistribution(cosTheta, roughness);
    pdf = D * cosTheta / 4.0f; // 半程向量PDF

    return H;
}

// 计算简化的粗糙度（基于metallic：1.0为0.05，0.0为0.5）
__device__ float getRoughnessFromMetallic(float metallic) {
    return glm::mix(0.5f, 0.05f, metallic);
}

