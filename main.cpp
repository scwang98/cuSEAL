
#include <iostream>

#include "sigma/kernelprovider.h"

int main() {

    sigma::KernelProvider::initialize();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
