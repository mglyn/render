#include "bvh.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <tuple>
#include <chrono>
#include <functional>

namespace bvh {

// 递归构建BVH（中值划分，按最长轴）
int buildRecursive(std::vector<BVHNode>& bvh, const std::vector<Triangle>& triangles, std::vector<int>& triIndices, int begin, int end, int maxLeafSize)
{
    BVHNode node; // 初始化
    AABB bounds;
    for (int i = begin; i < end; ++i)
        bounds.expandTri(triangles[triIndices[i]]);
    node.bounds = bounds;
    int n = end - begin;
    if (n <= maxLeafSize)
    {
        node.start = begin;
        node.count = n;
        int idx = (int)bvh.size();
        bvh.push_back(node);
        return idx;
    }
    glm::vec3 ext = bounds.extent();
    int axis = 0;
    if (ext.y > ext.x && ext.y >= ext.z)
        axis = 1;
    else if (ext.z > ext.x && ext.z >= ext.y)
        axis = 2;
    float midCoord = 0.f;
    for (int i = begin; i < end; ++i)
    {
        const Triangle &t = triangles[triIndices[i]];
        midCoord += (t.v0[axis] + t.v1[axis] + t.v2[axis]) / 3.f;
    }
    midCoord /= n;
    int pivot = std::partition(triIndices.begin() + begin, triIndices.begin() + end, [&](int triIdx)
                               { const Triangle &t=triangles[triIdx]; float c=(t.v0[axis]+t.v1[axis]+t.v2[axis])/3.f; return c < midCoord; }) -
                triIndices.begin();
    if (pivot == begin || pivot == end)
        pivot = begin + n / 2; // 退化处理
    int idx = (int)bvh.size();
    bvh.push_back(node);
    int left = buildRecursive(bvh, triangles, triIndices, begin, pivot, maxLeafSize);
    int right = buildRecursive(bvh, triangles, triIndices, pivot, end, maxLeafSize);
    bvh[idx].left = left;
    bvh[idx].right = right;
    return idx;
}

void build(std::vector<BVHNode>& bvh, const std::vector<Triangle>& triangles, std::vector<int>& triIndices, int maxLeafSize)
{
    bvh.clear();
    if (triangles.empty())
        return;
    
    // 记录构建开始时间
    auto start = std::chrono::high_resolution_clock::now();
    
    buildRecursive(bvh, triangles, triIndices, 0, (int)triangles.size(), maxLeafSize);
    
    // 记录构建结束时间
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 计算BVH统计信息
    int totalNodes = (int)bvh.size();
    int leafNodes = 0;
    int maxDepth = 0;
    int maxLeafTriangles = 0;
    float totalSurfaceArea = 0.0f;
    
    // 遍历所有节点收集统计
    std::function<void(int, int)> collectStats = [&](int nodeIdx, int depth) {
        if (nodeIdx < 0 || nodeIdx >= totalNodes) return;
        const BVHNode& node = bvh[nodeIdx];
        maxDepth = std::max(maxDepth, depth);
        totalSurfaceArea += node.bounds.surfaceArea();
        if (node.count > 0) {
            leafNodes++;
            maxLeafTriangles = std::max(maxLeafTriangles, node.count);
        } else {
            if (node.left >= 0) collectStats(node.left, depth + 1);
            if (node.right >= 0) collectStats(node.right, depth + 1);
        }
    };
    
    if (totalNodes > 0) {
        collectStats(0, 0);
    }
    
    // 打印性能和统计信息
    printf("\n=== BVH Build Performance & Statistics ===\n");
    printf("Triangles: %d\n", (int)triangles.size());
    printf("Build Time: %lld microseconds (%.2f ms)\n", 
           duration.count(), duration.count() / 1000.0);
    printf("Total Nodes: %d\n", totalNodes);
    printf("Leaf Nodes: %d\n", leafNodes);
    printf("Internal Nodes: %d\n", totalNodes - leafNodes);
    printf("Max Depth: %d\n", maxDepth);
    printf("Max Leaf Triangles: %d\n", maxLeafTriangles);
    printf("Total Surface Area: %.2f\n", totalSurfaceArea);
    
    if (leafNodes > 0) {
        printf("Avg Leaf Triangles: %.2f\n", (float)triangles.size() / leafNodes);
        float leafRatio = (float)leafNodes / totalNodes;
        printf("Leaf Ratio: %.2f%%\n", leafRatio * 100.0f);
        
        // 平衡因子 (理想平衡树的深度 vs 实际深度)
        float idealDepth = std::log2f((float)leafNodes);
        float balanceFactor = maxDepth / idealDepth;
        printf("Balance Factor: %.2f (1.0=perfect, higher=less balanced)\n", balanceFactor);
    }
    
    printf("==========================================\n\n");
}

} // namespace bvh
