#include "ui.h"
#include "renderer/cuda_path_tracing_renderer.h"
#include "scene/camera.h"
#include "application.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

namespace ui {

void init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

void shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

static void renderPathTracingSubmenu(CudaPathTracingRenderer* renderer, Camera& camera) {
    if (!renderer) return;

    ImGui::Separator();

    if (ImGui::TreeNode("Path Tracing Settings")) {
        int spp = renderer->samplesPerPixel;
        int maxDepth = renderer->maxDepth;
        bool enableDiffuseIS = renderer->enableDiffuseImportanceSampling;
        int lightingMode = static_cast<int>(renderer->lightingMode);

        ImGui::Text("Samples Per Pixel: %d", spp);
        ImGui::SameLine();
        if (ImGui::SmallButton("-##spp")) {
            if (spp > 1) {
                renderer->samplesPerPixel = spp - 1;
                camera.markDirty();
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("+##spp")) {
            renderer->samplesPerPixel = spp + 1;
            camera.markDirty();
        }

        ImGui::Text("Max Depth: %d", maxDepth);
        ImGui::SameLine();
        if (ImGui::SmallButton("-##depth")) {
            if (maxDepth > 1) {
                renderer->maxDepth = maxDepth - 1;
                camera.markDirty();
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("+##depth")) {
            renderer->maxDepth = maxDepth + 1;
            camera.markDirty();
        }

        ImGui::Separator();
        if (ImGui::Checkbox("Enable Diffuse Importance Sampling", &enableDiffuseIS)) {
            renderer->enableDiffuseImportanceSampling = enableDiffuseIS;
            camera.markDirty();
        }

        const char* lightingModeLabels[] = {
            "Direct Lighting",
            "Indirect Lighting",
            "MIS"
        };
        if (ImGui::Combo("Lighting Mode", &lightingMode, lightingModeLabels, IM_ARRAYSIZE(lightingModeLabels))) {
            renderer->lightingMode = static_cast<LightingMode>(lightingMode);
            camera.markDirty();
        }

        bool enableRussianRoulette = renderer->enableRussianRoulette;
        int rouletteStartDepth = renderer->rouletteStartDepth;
        int currMaxDepth = renderer->maxDepth;
        currMaxDepth = currMaxDepth > 1 ? currMaxDepth : 1;
        if (ImGui::TreeNodeEx("Russian Roulette", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Enable Russian Roulette", &enableRussianRoulette)) {
                renderer->enableRussianRoulette = enableRussianRoulette;
                camera.markDirty();
            }
            if (ImGui::SliderInt("Start Depth", &rouletteStartDepth, 1, currMaxDepth)) {
                renderer->rouletteStartDepth = rouletteStartDepth;
                camera.markDirty();
            }
            ImGui::TreePop();
        }

        ImGui::TreePop();
    }
}

static void renderResolutionSubmenu(Application* app) {
    // 渲染分辨率控制
    if (app) {
        ImGui::Separator();
        if (ImGui::TreeNode("Render Resolution")) {
            float renderScale = app->getRenderScale() * 100.0f; // 转换为百分比显示
            ImGui::Text("Render Size: %dx%d", app->getRenderWidth(), app->getRenderHeight());
            ImGui::Text("Display Size: %dx%d", app->getDisplayWidth(), app->getDisplayHeight());

            ImGui::Text("Render Scale: %.1f%%", renderScale);
            ImGui::SameLine();
            if(ImGui::SmallButton("+##Scale")) {
                if(renderScale < 100.0f) {
                    renderScale += 10.0f;
                    app->setRenderScale(renderScale / 100.0f);
                }
            }
            ImGui::SameLine();
            if(ImGui::SmallButton("-##Scale")) {
                if(renderScale > 10.0f) {
                    renderScale -= 10.0f;
                    app->setRenderScale(renderScale / 100.0f);
                }
            }
            
            ImGui::TreePop();
        }
    }
}

static void renderControlsSubmenu() {
    if (ImGui::TreeNodeEx("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BulletText("W/A/S/D - Move camera");
        ImGui::BulletText("Arrow Keys - Rotate view");
        ImGui::BulletText("Space/Shift - Up/Down");
        ImGui::BulletText("ESC - Exit");
        ImGui::TreePop();
    }
}

static void renderCameraSubmenu(Camera& camera) {
    if (ImGui::TreeNode("Camera Settings")) {
        glm::vec3 pos = camera.getPosition();
        float fov = camera.getFov();
        float yaw = camera.getYaw();
        float pitch = camera.getPitch();
        
        // 显示位置
        ImGui::Text("Position");
        ImGui::InputFloat3("##pos", &pos.x);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            camera.setPosition(pos);
        }
        
        // 显示姿态
        ImGui::Text("Yaw: %.1f, Pitch: %.1f", yaw, pitch);
        float yawPitch[2] = {yaw, pitch};
        ImGui::InputFloat2("##yawpitch", yawPitch);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            camera.setYawPitch(yawPitch[0], yawPitch[1]);
        }
        
        // 显示FOV
        ImGui::Text("FOV: %.1f", fov);
        ImGui::SameLine();
        if (ImGui::SmallButton("-##fov")) {
            if (fov > 10.0f) {
                camera.setFov(fov - 5.0f);
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("+##fov")) {
            if (fov < 120.0f) {
                camera.setFov(fov + 5.0f);
            }
        }
        
        ImGui::TreePop();
    }
}

static void renderModelsSubmenu(Scene* scene, Camera& camera) {
    if (ImGui::TreeNode("Models")) {
        // Add Model button
        if (ImGui::Button("Add Model")) {
            scene->addEmptyModel();
        }
        ImGui::SameLine();
        ImGui::Text("(%zu models)", scene->getModels().size());
        
        const auto& models = scene->getModels();
        for (size_t i = 0; i < models.size(); ++i) {
            const auto& model = models[i];
            ImGui::PushID(static_cast<int>(i));
            
            std::string label = model->getName().empty() ? "Model " + std::to_string(i) : model->getName();
            ImGui::AlignTextToFramePadding();
            bool nodeOpen = ImGui::TreeNode(label.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton(("del##delete" + std::to_string(i)).c_str())) {
                scene->removeModel(i);
                ImGui::PopID();
                continue; // Skip rendering this model since it's deleted
            }
            
            if (nodeOpen) {
                // Triangle count and Load OBJ button
                ImGui::Text("Triangles: %zu", model->triangles().size());
                
                // Static variable to track which model is showing the load dialog
                static int loadingModelIndex = -1;
                static char objPathInput[256] = "";
                
                if (ImGui::Button(("Load OBJ##load" + std::to_string(i)).c_str())) {
                    loadingModelIndex = static_cast<int>(i);
                    memset(objPathInput, 0, sizeof(objPathInput)); // Clear previous input
                }
                
                // Show input dialog if this model is selected for loading
                if (loadingModelIndex == static_cast<int>(i)) {
                    ImGui::InputText(("OBJ Path##path" + std::to_string(i)).c_str(), objPathInput, sizeof(objPathInput));
                    ImGui::SameLine();
                    if (ImGui::Button(("Load##confirm" + std::to_string(i)).c_str())) {
                        if (strlen(objPathInput) > 0) {
                            scene->loadObjToModel(i, std::string(objPathInput));
                        }
                        loadingModelIndex = -1; // Close dialog
                        memset(objPathInput, 0, sizeof(objPathInput));
                    }
                    ImGui::SameLine();
                    if (ImGui::Button(("Cancel##cancel" + std::to_string(i)).c_str())) {
                        loadingModelIndex = -1; // Close dialog
                        memset(objPathInput, 0, sizeof(objPathInput));
                    }
                }
                
                glm::vec3 pos = model->getPosition();
                glm::vec3 rot = model->getRotation();
                glm::vec3 scale = model->getScale();
                
                // 位置
                ImGui::Text("Position");
                ImGui::InputFloat3("##pos", &pos.x);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    model->setPosition(pos);
                    model->updateModelMatrix();
                    scene->setDirty();
                }
                
                // 旋转
                ImGui::Text("Rotation");
                ImGui::InputFloat3("##rot", &rot.x);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    model->setRotation(rot);
                    model->updateModelMatrix();
                    scene->setDirty();
                }
                
                // 缩放
                ImGui::Text("Scale");
                ImGui::InputFloat3("##scale", &scale.x);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    model->setScale(scale);
                    model->updateModelMatrix();
                    scene->setDirty();
                }
                
                // 材质信息
                ImGui::Separator();
                ImGui::Text("Material (Applied to all triangles)");
                
                // 获取当前默认材质作为编辑起点
                Material currentMat = model->material();
                glm::vec3 albedo = currentMat.albedo;
                float metallic = currentMat.metallic;
                
                bool materialChanged = false;
                
                // 颜色编辑
                ImGui::Text("Albedo (Color)");
                if (ImGui::ColorEdit3("##albedo", &albedo.x)) {
                    currentMat.albedo = albedo;
                    materialChanged = true;
                }
                
                // 金属度编辑
                ImGui::Text("Metallic");
                if (ImGui::SliderFloat("##metallic", &metallic, 0.0f, 1.0f, "%.3f")) {
                    currentMat.metallic = metallic;
                    materialChanged = true;
                }
                
                // RGB值显示
                ImGui::Text("RGB: (%.3f, %.3f, %.3f)", currentMat.albedo.r, currentMat.albedo.g, currentMat.albedo.b);
                
                // 应用材质更改
                if (materialChanged) {
                    model->setAllTrianglesMaterial(currentMat);
                    scene->setDirty();
                }
                
                // 自发光编辑
                ImGui::Text("Emission");
                glm::vec3 emission = model->emission();
                if (ImGui::ColorEdit3("##emission", &emission.x)) {
                    model->setEmission(emission);
                    scene->setDirty();
                }
                ImGui::Text("Intensity: (%.3f, %.3f, %.3f)", emission.r, emission.g, emission.b);
                
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
}

void renderUI(int& rendererMode, double fps, CudaPathTracingRenderer* renderer, Camera& camera, Scene* scene, Application* app) {
    ImGui::Begin("Renderer Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("FPS: %.1f", fps);

    renderPathTracingSubmenu(renderer, camera);

    ImGui::Spacing();
    ImGui::Separator();
    renderResolutionSubmenu(app);

    ImGui::Spacing();   
    ImGui::Separator();
    renderCameraSubmenu(camera);

    ImGui::Spacing();
    ImGui::Separator();
    renderModelsSubmenu(scene, camera);

    ImGui::Spacing();
    ImGui::Separator();
    renderControlsSubmenu();
    
    ImGui::End();
}

} // namespace ui
