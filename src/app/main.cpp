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
