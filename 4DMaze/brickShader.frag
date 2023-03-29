#version 330 core
in vec3 TexCoord;
in vec3 normal;
in vec3 fragLightPos;
in vec3 pos;

out vec4 FragColor;

uniform sampler3D tex;
uniform float proportion;

const float lightIntensity = 1.0f;

void main() {

    vec3 lightDir = normalize(fragLightPos - pos);
    float lightDistance = distance(fragLightPos, pos);
    float lightStrength = lightIntensity / (lightDistance * lightDistance);
    float diff = abs(clamp(lightStrength * dot(normal, lightDir), -1.0, 1.0));

	FragColor = diff * texture(tex, TexCoord);
}