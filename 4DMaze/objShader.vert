#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
  
out vec4 vertexColor;
out vec3 normal;
out vec3 fragLightPos;
out vec3 pos;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 lightPos;
uniform mat4 transform;

void main()
{
    vec4 transformPos =  transform * vec4(aPos, 1.0);
    gl_Position = projection * view * transformPos;

    vertexColor = vec4(0.5, 0.5, 0.8, 1.0);
    normal = aNormal;
    fragLightPos = lightPos;
    pos = transformPos.xyz;
}