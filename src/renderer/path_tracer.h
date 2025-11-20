#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <glm/glm.hpp>
#include <cstdint>
#include "renderer/pt_utils.h"

struct TriangleGPU;
struct MaterialGPU;
struct BVHNodeGPU;

void kernel_path_tracer(
    uint8_t* pbo_ptr, int width, int height,
    const glm::vec3& cam_pos, const glm::mat4& cam_view, float cam_fov,
    glm::vec3* accumulated_radiance, uint32_t frame_count,
    const TriangleGPU* triangles, int triangle_count,
    const MaterialGPU* materials,
    const BVHNodeGPU* bvh_nodes,
    const int* tri_indices,
    const int* light_indices, int light_count,
    int max_depth, int samples_per_pixel, bool enable_rr, int rr_depth,
    bool enable_diffuse_is, LightingMode lighting_mode
);

#endif // PATH_TRACER_H