#pragma once
#include <glm/glm.hpp>

class Camera {
public:
    Camera(const glm::vec3& position = glm::vec3(0.0f,0.0f,3.0f), 
        float yaw=-90.f, float pitch=0.f, float fov=60.f);

    void processMovement(bool forward, bool backward, bool left, bool right, bool up, bool down, float deltaTime, bool constrainToXZ = true);

    // 新增：统一处理旋转输入，内部用角速度
    void processRotationInput(bool left, bool right, bool up, bool down, float deltaTime);

    glm::vec3 getPosition() const { return position; }
    glm::vec3 getFront() const { return front; }
    glm::vec3 getRight() const { return right; }
    glm::vec3 getUp() const { return up; }
    float getFov() const { return fov; }

private:
    void updateVectors();
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp{0.0f,1.0f,0.0f};
    float yaw;
    float pitch;
    float fov;
    float movementSpeed = 3.0f;

    // 新增：角速度（度/秒）
    float rotationSpeed = 90.0f;
};
