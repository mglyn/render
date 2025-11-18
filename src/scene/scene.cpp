#include "scene.h"
#include <iostream>
#include <cuda_runtime.h>

Scene::Scene() {
    // 构造函数，脏标记默认为true
}

Scene::~Scene() {
    // 析构函数，目前为空
}

void Scene::addShape(const Shape& shape) {
    shapes_.push_back(shape);
    setDirty(); // 添加物体后设置脏标记
}

void Scene::addModel(std::unique_ptr<Model> model) {
    if (!model || model->empty()) {
        std::cerr << "[Scene] 添加模型失败：模型为空" << std::endl;
        return;
    }
    // 不再将模型三角形添加到Shape列表，避免重复存储
    // 模型将通过BVH系统单独管理
    models_.push_back(std::move(model));
    setDirty();
}

bool Scene::addModelFromObj(const std::string &path, const MaterialPOD &mat) {
    auto model = std::make_unique<Model>(mat);
    if (!model->loadObj(path, mat)) {
        std::cerr << "[Scene] OBJ 加载失败: " << path << std::endl;
        return false;
    }
    model->buildBVH();
    addModel(std::move(model));
    return true;
}

bool Scene::uploadBVHToGPU() {
    freeBVHGPU();
    gpuModels_.clear();
    
    for (const auto& model : models_) {
        if (model->empty()) continue;
        
        ModelGPU gpuModel{};
        const auto& bvh = model->bvh();
        const auto& triangles = model->triangles();
        
        // 上传BVH节点
        gpuModel.nodeCount = static_cast<int>(bvh.size());
        if (gpuModel.nodeCount > 0) {
            std::vector<BVHNodeGPU> gpuNodes(gpuModel.nodeCount);
            for (int i = 0; i < gpuModel.nodeCount; ++i) {
                const auto& node = bvh[i];
                gpuNodes[i].minBounds = node.bounds.min;
                gpuNodes[i].maxBounds = node.bounds.max;
                gpuNodes[i].left = node.left;
                gpuNodes[i].right = node.right;
                gpuNodes[i].start = node.start;
                gpuNodes[i].count = node.count;
            }
            
            cudaError_t err = cudaMalloc(&gpuModel.bvhNodes, gpuModel.nodeCount * sizeof(BVHNodeGPU));
            if (err != cudaSuccess) return false;
            err = cudaMemcpy(gpuModel.bvhNodes, gpuNodes.data(), gpuModel.nodeCount * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) return false;
        }
        
        // 上传三角形索引（用于BVH叶子节点）
        const auto& triIndices = model->getTriangleIndices();
        if (!triIndices.empty()) {
            cudaError_t err = cudaMalloc(&gpuModel.triangleIndices, triIndices.size() * sizeof(int));
            if (err != cudaSuccess) return false;
            err = cudaMemcpy(gpuModel.triangleIndices, triIndices.data(), triIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) return false;
        }
        
        // 上传三角形
        gpuModel.triangleCount = static_cast<int>(triangles.size());
        if (gpuModel.triangleCount > 0) {
            cudaError_t err = cudaMalloc(&gpuModel.triangles, gpuModel.triangleCount * sizeof(TrianglePOD));
            if (err != cudaSuccess) return false;
            err = cudaMemcpy(gpuModel.triangles, triangles.data(), gpuModel.triangleCount * sizeof(TrianglePOD), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) return false;
        }
        
        gpuModels_.push_back(gpuModel);
    }
    
    bvhUploaded_ = true;
    return true;
}

void Scene::freeBVHGPU() {
    for (auto& gpuModel : gpuModels_) {
        if (gpuModel.bvhNodes) cudaFree(gpuModel.bvhNodes);
        if (gpuModel.triangleIndices) cudaFree(gpuModel.triangleIndices);
        if (gpuModel.triangles) cudaFree(gpuModel.triangles);
    }
    gpuModels_.clear();
    bvhUploaded_ = false;
}