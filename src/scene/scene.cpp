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

// 新的 addModel 实现
void Scene::addModel(std::unique_ptr<Model> model, const glm::vec3 &pos, const glm::vec3 &rotation, const glm::vec3 &scale) {
    if (!model || model->empty()) {
        std::cerr << "[Scene] 添加模型失败：模型为空" << std::endl;
        return;
    }
    model->setPosition(pos);
    model->setRotation(rotation);
    model->setScale(scale);
    model->updateModelMatrix(); // 在所有变换设置后，手动更新一次矩阵
    model->buildBVH();
    models_.push_back(std::move(model));
    setDirty();
}

// 新的静态方法 createModelFromObj
std::unique_ptr<Model> Scene::createModelFromObj(const std::string &path, const Material &mat) {
    auto model = std::make_unique<Model>(mat);
    if (!model->loadObj(path, mat)) {
        std::cerr << "[Scene] OBJ 加载失败: " << path << std::endl;
        return nullptr;
    }
    return model;
}

bool Scene::uploadBVHToGPU() {
    // 1. 创建一个临时的 ModelGPU 向量
    std::vector<ModelGPU> newGpuModels;

    for (const auto& model : models_) {
        if (model->empty()) continue;
        
        ModelGPU gpuModel{};
        const auto& bvh = model->bvh();
        
        // 应用模型变换到三角形
        std::vector<TrianglePOD> worldTriangles = model->triangles();
        const glm::mat4& modelMatrix = model->getModelMatrix();
        for (auto& tri : worldTriangles) {
            tri.v0 = glm::vec3(modelMatrix * glm::vec4(tri.v0, 1.0f));
            tri.v1 = glm::vec3(modelMatrix * glm::vec4(tri.v1, 1.0f));
            tri.v2 = glm::vec3(modelMatrix * glm::vec4(tri.v2, 1.0f));
        }

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
            if (err != cudaSuccess) { /* 错误处理 */ return false; }
            err = cudaMemcpy(gpuModel.bvhNodes, gpuNodes.data(), gpuModel.nodeCount * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { /* 错误处理 */ return false; }
        }
        
        // 上传三角形索引
        const auto& triIndices = model->getTriangleIndices();
        if (!triIndices.empty()) {
            cudaError_t err = cudaMalloc(&gpuModel.triangleIndices, triIndices.size() * sizeof(int));
            if (err != cudaSuccess) { /* 错误处理 */ return false; }
            err = cudaMemcpy(gpuModel.triangleIndices, triIndices.data(), triIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { /* 错误处理 */ return false; }
        }
        
        // 上传变换后的三角形
        gpuModel.triangleCount = static_cast<int>(worldTriangles.size());
        if (gpuModel.triangleCount > 0) {
            cudaError_t err = cudaMalloc(&gpuModel.triangles, gpuModel.triangleCount * sizeof(TrianglePOD));
            if (err != cudaSuccess) { /* 错误处理 */ return false; }
            err = cudaMemcpy(gpuModel.triangles, worldTriangles.data(), gpuModel.triangleCount * sizeof(TrianglePOD), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { /* 错误处理 */ return false; }
        }
        
        newGpuModels.push_back(gpuModel);
    }
    
    // 2. 释放旧的 GPU 资源
    freeBVHGPU();

    // 3. 用新的数据交换旧的向量，这是一个原子操作
    gpuModels_.swap(newGpuModels);
    
    bvhUploaded_ = true;
    return true;
}

void Scene::freeBVHGPU() {
    for (auto& gpuModel : gpuModels_) {
        if (gpuModel.bvhNodes) cudaFree(gpuModel.bvhNodes);
        if (gpuModel.triangleIndices) cudaFree(gpuModel.triangleIndices);
        if (gpuModel.triangles) cudaFree(gpuModel.triangles);
    }
    gpuModels_.clear(); // 在这里清空是安全的，因为它持有的是旧数据
    bvhUploaded_ = false;
}