#include "ui.h"
#include "renderer/cuda_path_tracing_renderer.h"
#include "scene/camera.h"

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

static void renderControlsSubmenu() {
    if (ImGui::TreeNodeEx("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BulletText("W/A/S/D - Move camera");
        ImGui::BulletText("Arrow Keys - Rotate view");
        ImGui::BulletText("Space/Shift - Up/Down");
        ImGui::BulletText("ESC - Exit");
        ImGui::TreePop();
    }
}

void renderUI(int& rendererMode, double fps, CudaPathTracingRenderer* renderer, Camera& camera) {
    ImGui::Begin("Renderer Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("FPS: %.1f", fps);

    renderPathTracingSubmenu(renderer, camera);

    static const char* rendererNames[] = { "Path Tracing" };
    (void)rendererNames; // currently unused
    
    ImGui::Spacing();
    ImGui::Separator();
    renderControlsSubmenu();
    
    ImGui::End();
}

} // namespace ui
