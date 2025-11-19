#include "ui.h"
#include "renderer/cuda_path_tracing_renderer.h"

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

static void renderPathTracingSubmenu(CudaPathTracingRenderer* renderer) {
    if (!renderer) return;

    ImGui::Separator();

    if (ImGui::TreeNode("Path Tracing Settings")) {
        int spp = renderer->getSpp();
        int maxDepth = renderer->getMaxDepth();
        bool enableDiffuseIS = renderer->getEnableDiffuseImportanceSampling();
        int lightingMode = static_cast<int>(renderer->getLightingMode());

        ImGui::Text("Samples Per Pixel: %d", spp);
        ImGui::SameLine();
        if (ImGui::SmallButton("-##spp")) {
            if (spp > 1) {
                renderer->setSpp(spp - 1);
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("+##spp")) {
            renderer->setSpp(spp + 1);
        }

        ImGui::Text("Max Depth: %d", maxDepth);
        ImGui::SameLine();
        if (ImGui::SmallButton("-##depth")) {
            if (maxDepth > 1) {
                renderer->setMaxDepth(maxDepth - 1);
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("+##depth")) {
            renderer->setMaxDepth(maxDepth + 1);
        }

        ImGui::Separator();
        const char* lightingModeItems[] = { "Direct Only", "Indirect Only", "Full (MIS)" };
        if (ImGui::Combo("Lighting Mode", &lightingMode, lightingModeItems, IM_ARRAYSIZE(lightingModeItems))) {
            renderer->setLightingMode(static_cast<LightingMode>(lightingMode));
        }
        if (ImGui::Checkbox("Enable Diffuse Importance Sampling", &enableDiffuseIS)) {
            renderer->setEnableDiffuseImportanceSampling(enableDiffuseIS);
        }

        ImGui::TreePop();
    }
}

static void renderControlsSubmenu() {
    if (ImGui::TreeNodeEx("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BulletText("W/A/S/D - Move camera");
        ImGui::BulletText("Arrow Keys - Rotate view");
        ImGui::BulletText("Space/Shift - Up/Down");
        ImGui::BulletText("T - Add random sphere");
        ImGui::BulletText("ESC - Exit");
        ImGui::TreePop();
    }
}

void renderUI(int& rendererMode, double fps, CudaPathTracingRenderer* renderer) {
    ImGui::Begin("Renderer Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("FPS: %.1f", fps);

    renderPathTracingSubmenu(renderer);

    static const char* rendererNames[] = { "Path Tracing" };
    (void)rendererNames; // currently unused
    
    ImGui::Spacing();
    ImGui::Separator();
    renderControlsSubmenu();
    
    ImGui::End();
}

} // namespace ui
