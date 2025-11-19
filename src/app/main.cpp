#include "application.h"
#include <iostream>

int main() {
    Application app(1280, 720);
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }

    app.run();
    app.shutdown();

    return 0;
}
// 给ui添加下级菜单 控制pathrtacing的各种参数
// -路径追踪设置菜单：
// --单像素采样数菜单
// --路径最大深度菜单
// --向光源采样菜单
// --余弦加权半球采样菜单
// --俄罗斯轮盘赌菜单