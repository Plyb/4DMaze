#version 330 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec3 aTexCoord;
layout (location = 2) in vec3 aNormal;

out vec3 TexCoord;
out vec3 normal;
out vec3 fragLightPos;
out vec3 pos;

uniform vec4 cameraPos;
uniform mat4 dropDimension;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 lightPos;

void main() {
    vec4 pos4 = (dropDimension * (aPos - cameraPos) +
		vec4(cameraPos.xy, dropDimension[2][2] * cameraPos.z + dropDimension[3][2] * cameraPos.w, 1.0f));
    gl_Position = projection * view *  pos4;
	
	TexCoord = aTexCoord;
    normal = aNormal;
    fragLightPos = lightPos;
    pos = pos4.xyz;
}