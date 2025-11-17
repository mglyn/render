#pragma once
#include <vector>
#include "shape.h"

// Scene 类：统一管理场景中的所有物体（Shape）
// 后续可扩展为管理材质、灯光、加速结构等
class Scene {
public:
    Scene();
    ~Scene();

    // 添加一个物体到场景，并标记为已修改
    void addShape(const Shape& shape);

    // 获取场景中所有物体的只读引用
    const std::vector<Shape>& getShapes() const { return shapes_; }

    // 获取场景中所有物体的可写引用，并标记为已修改
    std::vector<Shape>& getShapes() { setDirty(); return shapes_; }

    // 检查场景是否被修改过
    bool isDirty() const { return dirty_; }

    // 清除修改标记
    void clearDirty() { dirty_ = false; }

    // 手动设置修改标记
    void setDirty() { dirty_ = true; }

private:
    std::vector<Shape> shapes_;
    bool dirty_ = true; // 脏标记，默认为true，确保初始场景被上传
};
