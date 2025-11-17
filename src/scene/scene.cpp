#include "scene.h"

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
