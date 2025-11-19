#pragma once
#include <glm/glm.hpp>

class Camera {
public:
    Camera(const glm::vec3& position = glm::vec3(0.0f,0.0f,3.0f), 
        float yaw=-90.f, float pitch=0.f, float fov=60.f);

    void processMovement(bool forward, bool backward, bool left, bool right, bool up, bool down, float deltaTime, bool constrainToXZ = true);
    void processRotationInput(bool left, bool right, bool up, bool down, float deltaTime);

    glm::vec3 getPosition() const { return position; }
    glm::vec3 getFront() const { return front; }
    glm::vec3 getRight() const { return right; }
    glm::vec3 getUp() const { return up; }
    float getFov() const { return fov; }

    void setPosition(const glm::vec3& pos) { 
        position = pos;
        updateVectors(); 
        markDirty();
    }
    void setFov(float newFov) { 
        fov = newFov; 
        markDirty(); 
    }
    void setYawPitch(float newYaw, float newPitch) { 
        yaw = newYaw; pitch = newPitch; 
        updateVectors(); 
        markDirty(); 
    }

    // 获取视图和投影矩阵
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;

    bool isDirty() const { return dirty; }
    void clearDirty() { dirty = false; }
    void markDirty() { dirty = true; }

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

    float movementSpeed = 1.0f;
    float rotationSpeed = 30.0f;

    bool dirty = true;
};
