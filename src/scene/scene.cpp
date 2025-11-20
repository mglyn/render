#include "scene.h"
#include "bvh/bvh.h"
#include <iostream>
#include <cuda_runtime.h>
#include <map>

// Helper to convert CPU material to GPU material
static MaterialGPU toMaterialGPU(const Material& mat) {
    MaterialGPU gpuMat;
    gpuMat.albedo = mat.albedo;
    gpuMat.metallic = mat.metallic;
    // gpuMat.emission = mat.emission; // Assuming emission is handled separately or part of Material
    return gpuMat;
}

bool Scene::buildAndUploadScene() {
    if (!isDirty()) {
        return true; // No changes, no need to rebuild
    }

    std::cout << "Scene is dirty, rebuilding and uploading to GPU..." << std::endl;

    freeSceneGPU(); // Free old data before building

    // Clear host containers
    host_triangles_.clear();
    host_materials_.clear();
    host_tri_indices_.clear();
    host_light_indices_.clear();

    std::map<Material, int> materialMap;
    int materialCounter = 0;

    // 1. Consolidate all triangles and materials from all models
    std::cout << "Processing " << models_.size() << " models..." << std::endl;
    for (const auto& model : models_) {
        if (model->empty()) {
            std::cout << "  - Skipping empty model." << std::endl;
            continue;
        }
        std::cout << "  - Processing model with " << model->triangles().size() << " triangles." << std::endl;

        const auto& modelMatrix = model->getModelMatrix();
        const auto& normalMatrix = glm::transpose(glm::inverse(modelMatrix));
        const auto& triangles = model->triangles();
        const auto& materials = model->materials();
        const auto& triMatIndices = model->triangleMaterialIndices();

        for (size_t i = 0; i < triangles.size(); ++i) {
            const auto& tri = triangles[i];
            int matCPUIndex = (i < triMatIndices.size()) ? triMatIndices[i] : 0;
            const Material& mat = (matCPUIndex < materials.size()) ? materials[matCPUIndex] : Material();

        // Get or create material index
        int finalMaterialIndex;
        if (materialMap.find(mat) == materialMap.end()) {
            finalMaterialIndex = materialCounter++;
            materialMap[mat] = finalMaterialIndex;
            host_materials_.push_back(toMaterialGPU(mat));
            host_materials_.back().emission = model->emission(); // Set emission for the whole model's materials
        } else {
            finalMaterialIndex = materialMap[mat];
        }            TriangleGPU gpuTri;
            // Apply transform
            gpuTri.v0 = glm::vec3(modelMatrix * glm::vec4(tri.v0, 1.0f));
            gpuTri.v1 = glm::vec3(modelMatrix * glm::vec4(tri.v1, 1.0f));
            gpuTri.v2 = glm::vec3(modelMatrix * glm::vec4(tri.v2, 1.0f));
            gpuTri.n0 = glm::normalize(glm::vec3(normalMatrix * glm::vec4(tri.n0, 0.0f)));
            gpuTri.n1 = glm::normalize(glm::vec3(normalMatrix * glm::vec4(tri.n1, 0.0f)));
            gpuTri.n2 = glm::normalize(glm::vec3(normalMatrix * glm::vec4(tri.n2, 0.0f)));
            gpuTri.t0 = tri.t0;
            gpuTri.t1 = tri.t1;
            gpuTri.t2 = tri.t2;
            gpuTri.materialIndex = finalMaterialIndex;

            host_triangles_.push_back(gpuTri);

            // Check for lights
            if (glm::dot(model->emission(), model->emission()) > 0.0f) {
                host_light_indices_.push_back(host_triangles_.size() - 1);
            }
        }
    }

    std::cout << "Consolidated " << host_triangles_.size() << " triangles and " << host_materials_.size() << " materials." << std::endl;
    if(host_light_indices_.empty()){
        std::cout << "Warning: No light sources found in the scene." << std::endl;
    } else {
        std::cout << "Found " << host_light_indices_.size() << " light triangles." << std::endl;
    }

    // 2. Build scene-level BVH
    if (!host_triangles_.empty()) {
        std::vector<Triangle> temp_triangles_for_bvh;
        temp_triangles_for_bvh.reserve(host_triangles_.size());
        for(const auto& t_gpu : host_triangles_){
            Triangle t;
            t.v0 = t_gpu.v0; t.v1 = t_gpu.v1; t.v2 = t_gpu.v2;
            temp_triangles_for_bvh.push_back(t);
        }
        
        host_tri_indices_.resize(host_triangles_.size());
        for(size_t i = 0; i < host_triangles_.size(); ++i) host_tri_indices_[i] = i;

        bvh::build(host_bvh_nodes_, temp_triangles_for_bvh, host_tri_indices_, 4);
        std::cout << "Built scene BVH with " << host_bvh_nodes_.size() << " nodes." << std::endl;
    }

    // 3. Upload to GPU
    cudaError_t err;

    std::cout << "Uploading data to GPU..." << std::endl;

    if (!host_triangles_.empty()) {
        err = cudaMalloc(&d_triangles_, host_triangles_.size() * sizeof(TriangleGPU));
        if (err != cudaSuccess) return false;
        err = cudaMemcpy(d_triangles_, host_triangles_.data(), host_triangles_.size() * sizeof(TriangleGPU), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "  - Triangle upload failed: " << cudaGetErrorString(err) << std::endl; return false; }
    }

    if (!host_materials_.empty()) {
        err = cudaMalloc(&d_materials_, host_materials_.size() * sizeof(MaterialGPU));
        if (err != cudaSuccess) return false;
        err = cudaMemcpy(d_materials_, host_materials_.data(), host_materials_.size() * sizeof(MaterialGPU), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "  - Material upload failed: " << cudaGetErrorString(err) << std::endl; return false; }
    }

    if (!host_bvh_nodes_.empty()) {
        std::vector<BVHNodeGPU> gpu_nodes(host_bvh_nodes_.size());
        for(size_t i=0; i<host_bvh_nodes_.size(); ++i) {
            gpu_nodes[i].minBounds = host_bvh_nodes_[i].bounds.min;
            gpu_nodes[i].maxBounds = host_bvh_nodes_[i].bounds.max;
            gpu_nodes[i].left = host_bvh_nodes_[i].left;
            gpu_nodes[i].right = host_bvh_nodes_[i].right;
            gpu_nodes[i].start = host_bvh_nodes_[i].start;
            gpu_nodes[i].count = host_bvh_nodes_[i].count;
        }
        err = cudaMalloc(&d_bvhNodes_, gpu_nodes.size() * sizeof(BVHNodeGPU));
        if (err != cudaSuccess) { std::cerr << "  - BVH node upload failed: " << cudaGetErrorString(err) << std::endl; return false; }
        err = cudaMemcpy(d_bvhNodes_, gpu_nodes.data(), gpu_nodes.size() * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "  - BVH node memcpy failed: " << cudaGetErrorString(err) << std::endl; return false; }
    }

    if (!host_tri_indices_.empty()) {
        err = cudaMalloc(&d_triIndices_, host_tri_indices_.size() * sizeof(int));
        if (err != cudaSuccess) return false;
        err = cudaMemcpy(d_triIndices_, host_tri_indices_.data(), host_tri_indices_.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "  - Triangle index upload failed: " << cudaGetErrorString(err) << std::endl; return false; }
    }

    if (!host_light_indices_.empty()) {
        err = cudaMalloc(&d_lightIndices_, host_light_indices_.size() * sizeof(int));
        if (err != cudaSuccess) return false;
        err = cudaMemcpy(d_lightIndices_, host_light_indices_.data(), host_light_indices_.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { std::cerr << "  - Light index upload failed: " << cudaGetErrorString(err) << std::endl; return false; }
        std::cout << "Uploaded " << host_light_indices_.size() << " light triangles." << std::endl;
    }

    std::cout << "GPU upload complete." << std::endl;

    setDirty(false); // Mark as clean
    return true;
}

void Scene::freeSceneGPU() {
    cudaFree(d_triangles_); d_triangles_ = nullptr;
    cudaFree(d_materials_); d_materials_ = nullptr;
    cudaFree(d_bvhNodes_); d_bvhNodes_ = nullptr;
    cudaFree(d_triIndices_); d_triIndices_ = nullptr;
    cudaFree(d_lightIndices_); d_lightIndices_ = nullptr;
}

Scene::~Scene() {
    freeSceneGPU();
}

// Keep addModel and createModelFromObj, but remove old GPU upload logic
void Scene::addModel(std::unique_ptr<Model> model, const glm::vec3 &pos, const glm::vec3 &rotation, const glm::vec3 &scale) {
    if (!model || model->empty()) {
        std::cerr << "[Scene] Add model failed: model is empty" << std::endl;
        return;
    }
    model->setPosition(pos);
    model->setRotation(rotation);
    model->setScale(scale);
    model->updateModelMatrix();
    // We no longer build BVH per model
    models_.push_back(std::move(model));
    setDirty();
}

std::unique_ptr<Model> Scene::createModelFromObj(const std::string &path, const Material &mat) {
    auto model = std::make_unique<Model>(mat);
    if (!model->loadObj(path, mat)) {
        std::cerr << "[Scene] OBJ load failed: " << path << std::endl;
        return nullptr;
    }
    return model;
}

Scene::Scene() : dirty_(true) {}