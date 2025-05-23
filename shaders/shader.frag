#version 450

layout (binding = 0) uniform sampler2D inputImage;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(inputImage, uv);
}