#include "application.h"
#include <iostream>

int main() {
    Application app(1600, 900);
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }

    app.run();
    app.shutdown();

    return 0;
}
