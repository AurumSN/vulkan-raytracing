#include "window.h"

#include <stdexcept>

Window::Window(int width, int height, std::string name) : _width{ width }, _height{ height }, _name{ name } {
    SDL_Init(SDL_INIT_EVERYTHING);

    _window = SDL_CreateWindow("SDF Viewer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _width, _height, SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN);
    _running = true;
}

Window::~Window() {
    SDL_DestroyWindow(_window);
    SDL_Quit();
}

bool Window::shouldClose() {
    return !_running;
}

VkExtent2D Window::getExtent() {
    return { static_cast<uint32_t>(_width), static_cast<uint32_t>(_height) };
}

void Window::createWindowSurface(VkInstance instance, VkSurfaceKHR *surface) {
    if (SDL_Vulkan_CreateSurface(_window, instance, surface) != SDL_TRUE) {
        throw std::runtime_error("failed to create window surface");
    }
}