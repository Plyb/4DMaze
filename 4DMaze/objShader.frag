#version 330 core
out vec4 FragColor;
  
in vec4 vertexColor;
in vec3 normal;
in vec3 fragLightPos;
in vec3 pos;

const float lightIntensity = 1.0f;

void main()
{
    vec3 lightDir = normalize(fragLightPos - pos);
    float lightDistance = distance(fragLightPos, pos);
    float lightStrength = lightIntensity / (lightDistance * lightDistance);
    float diff = clamp(lightStrength * dot(normal, lightDir), 0.0, 1.0);

    FragColor = diff * vertexColor;
} 