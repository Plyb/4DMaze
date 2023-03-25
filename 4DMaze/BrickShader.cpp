#pragma once
#include "Camera.h"
#include "CameraViewShader.h"
#include <string>
#include <iostream>
#include "BrickTexture.cpp"
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

class BrickShader : public CameraViewShader {
public:
	BrickShader() : 
		CameraViewShader("./brickShader.vert", "./brickShader.frag")
	{
		loadBrickTexture();
	}

	void update(Camera& camera) {
		CameraViewShader::update(camera);
		setMatrix("dropDimension", createDropDimensionMat(camera));
		setFloat("cameraPos", camera.pos.x, camera.pos.y, camera.pos.z, camera.pos.w);
	}

private:
	glm::mat4 createDropDimensionMat(Camera& camera) {
		float testMat4[16] = {
			1.0f, 0.0f, 0.0f,					0.0f,
			0.0f, 1.0f, 0.0f,					0.0f,
			0.0f, 0.0f, cos(glm::radians(camera.psi)), 0.0f,
			0.0f, 0.0f, sin(glm::radians(camera.psi)), 0.0f
		};
		return glm::make_mat4(testMat4);
	}

	int loadBrickTexture() {
		unsigned int texture;
		glGenTextures(1, &texture);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, texture);

		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		BrickTexture brickTexture;
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, brickTexture.sideLength, brickTexture.sideLength, brickTexture.sideLength, 0, GL_RGBA, GL_UNSIGNED_BYTE, brickTexture.pixels);
		glGenerateMipmap(GL_TEXTURE_3D);

		glBindTexture(GL_TEXTURE_3D, texture);
		return texture;
	}
};