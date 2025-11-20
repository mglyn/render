#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>

class Shader {
public:
    Shader();
    ~Shader();

    // 从文件加载、编译和链接顶点和片段着色器
    bool load(const std::string& vertexPath, const std::string& fragmentPath);

    // 使用着色器程序
    void use() const;

    // 设置uniform变量
    void setVec2(const std::string& name, const glm::vec2& value);

    // 获取着色器程序ID
    GLuint getProgramID() const { return _programID; }

private:
    GLuint _programID;

    // 检查编译或链接错误的辅助函数
    bool checkCompileErrors(GLuint shader, const std::string& type);
};

#endif // SHADER_H
