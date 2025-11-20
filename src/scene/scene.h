#pragma once
#include <vector>
#include <memory>
#include <string>

#include "struct/shape.h"
#include "struct/modelGpu.h"
#include "struct/shapeGpu.h"
#include "struct/materialGpu.h"
#include "bvh/bvh.h"
#include "scene/model.h"

class Scene {
public:
    Scene();
    ~Scene();

    void addModel(std::unique_ptr<Model> model, const glm::vec3 &pos, const glm::vec3 &rotation, const glm::vec3 &scale);
    static std::unique_ptr<Model> createModelFromObj(const std::string &path, const Material &mat);

    bool isDirty() const { return dirty_; }
    void setDirty(bool dirty = true) { dirty_ = dirty; }

    // New unified scene build and access methods
    bool buildAndUploadScene();
    void freeSceneGPU();

    const TriangleGPU* getTrianglesGPU() const { return d_triangles_; }
    const MaterialGPU* getMaterialsGPU() const { return d_materials_; }
    const BVHNodeGPU* getBvhNodesGPU() const { return d_bvhNodes_; }
    const int* getTriangleIndicesGPU() const { return d_triIndices_; }
    const int* getLightIndicesGPU() const { return d_lightIndices_; }
    int getTriangleCount() const { return host_triangles_.size(); }
    int getLightCount() const { return host_light_indices_.size(); }

private:
    // CPU data
    std::vector<std::unique_ptr<Model>> models_;
    bool dirty_ = true;

    // Unified scene data on CPU (for building)
    std::vector<TriangleGPU> host_triangles_;
    std::vector<MaterialGPU> host_materials_;
    std::vector<BVHNode> host_bvh_nodes_;
    std::vector<int> host_tri_indices_;
    std::vector<int> host_light_indices_;

    // Unified scene data on GPU (device pointers)
    TriangleGPU* d_triangles_ = nullptr;
    MaterialGPU* d_materials_ = nullptr;
    BVHNodeGPU* d_bvhNodes_ = nullptr;
    int* d_triIndices_ = nullptr;
    int* d_lightIndices_ = nullptr;
};
