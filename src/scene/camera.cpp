#include "camera.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(const glm::vec3& position_, float yaw_, float pitch_, float fov_)
    : position(position_), yaw(yaw_), pitch(pitch_), fov(fov_) { updateVectors(); }

void Camera::processMovement(bool forward, bool backward, bool left, bool rightB, bool upB, bool down, float deltaTime, bool constrainToXZ){
    float velocity = movementSpeed * deltaTime;
    if(constrainToXZ){
        glm::vec3 fwd = glm::normalize(glm::vec3(front.x, 0.0f, front.z));
        glm::vec3 rgt = glm::normalize(glm::vec3(right.x, 0.0f, right.z));
        if(forward) position += fwd * velocity;
        if(backward) position -= fwd * velocity;
        if(left) position -= rgt * velocity;
        if(rightB) position += rgt * velocity;
    } else {
        if(forward) position += front * velocity;
        if(backward) position -= front * velocity;
        if(left) position -= right * velocity;
        if(rightB) position += right * velocity;
    }
    if(upB) position += worldUp * velocity;
    if(down) position -= worldUp * velocity;

    markDirty();
}


// 统一处理旋转输入，类似于位置移动
void Camera::processRotationInput(bool left, bool right, bool up, bool down, float deltaTime) {
    float yawDelta = 0.0f, pitchDelta = 0.0f;
    if (left) yawDelta -= rotationSpeed * deltaTime;
    if (right) yawDelta += rotationSpeed * deltaTime;
    if (up) pitchDelta += rotationSpeed * deltaTime;
    if (down) pitchDelta -= rotationSpeed * deltaTime;
    yaw += yawDelta;
    pitch += pitchDelta;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    updateVectors();

    markDirty();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getProjectionMatrix(float aspectRatio) const {
    return glm::perspective(glm::radians(fov), aspectRatio, 0.1f, 100.0f);
}

void Camera::updateVectors(){
    glm::vec3 f;
    f.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    f.y = sin(glm::radians(pitch));
    f.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(f);
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}
