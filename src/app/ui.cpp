#include "ui.h"

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

void renderUI(int& rendererMode) {
    ImGui::Begin("Renderer Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("Select Rendering Mode:");
    ImGui::Separator();
    
    static const char* rendererNames[] = { "Path Tracing", "Wireframe" };
    if (ImGui::Combo("Renderer##combo", &rendererMode, rendererNames, IM_ARRAYSIZE(rendererNames))) {
        ImGui::OpenPopup("RendererChangePopup");
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::BulletText("W/A/S/D - Move camera");
    ImGui::BulletText("Arrow Keys - Rotate view");
    ImGui::BulletText("Space/Shift - Up/Down");
    ImGui::BulletText("T - Add random sphere");
    ImGui::BulletText("ESC - Exit");
    
    ImGui::End();
}

} // namespace ui
