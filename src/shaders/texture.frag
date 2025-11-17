#version 330 core
in vec2 vTex;
out vec4 FragColor;
uniform sampler2D uTexture;
void main() {
    vec3 c = texture(uTexture, vTex).rgb;
    // simple gamma
    c = pow(c, vec3(1.0/2.2));
    FragColor = vec4(c, 1.0);
}
