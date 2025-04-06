#pragma once

#include <vulkan/vulkan.h>
#include <SDL2/SDL_keycode.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <string>

class Window {
    SDL_Window* _window;

    const int _width;
    const int _height;
    std::string _name;
    bool _running;

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

public:
    Window(int width, int height, std::string name);
    ~Window();

    bool shouldClose();

    SDL_Window* window() {
        return _window;
    }

    void stopWindow() {
        _running = false;
    }
    
    VkExtent2D getExtent();
    
    void createWindowSurface(VkInstance instance, VkSurfaceKHR *surface);
};