#include "window.h"

#include <stdexcept>

Window::Window(int width, int height, std::string name) : _width{ width }, _height{ height }, _name{ name } {
    SDL_Init(SDL_INIT_EVERYTHING);

    _window = SDL_CreateWindow("SDF Viewer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _width, _height, SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN);
    _running = true;
    _renderer = SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED);
    _texture = SDL_CreateTexture(
        _renderer,
        SDL_PIXELFORMAT_ABGR8888,
        SDL_TEXTUREACCESS_STREAMING,
        _width,
        _height);
}

Window::~Window() {
    SDL_DestroyWindow(_window);
    SDL_Quit();
}

bool Window::shouldClose() {
    return !_running;
}

void Window::createWindowSurface(VkInstance instance, VkSurfaceKHR *surface) {
    return;
    if (SDL_Vulkan_CreateSurface(_window, instance, surface) != SDL_TRUE) {
        throw std::runtime_error("failed to create window surface");
    }
}

void Window::draw(const uint32_t *pixels) {
    SDL_UpdateTexture(_texture, nullptr, pixels, _width * sizeof(uint32_t));
    SDL_RenderClear(_renderer);
    SDL_RenderCopy(_renderer, _texture, nullptr, nullptr);
    SDL_RenderPresent(_renderer);
}