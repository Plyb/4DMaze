#pragma once
#include "Camera.h"
#include "Shader.h"
#include <glm/gtc/type_ptr.hpp>

class CameraViewShader : public Shader {
public:
	CameraViewShader(const char* vertexPath, const char* fragmentPath) :
		 Shader(vertexPath, fragmentPath) {}

	virtual void update(Camera& camera) {
		setMatrix("view", createViewTransform(camera));
		setMatrix("projection", createProjectionTransform());
		glm::vec3 threeCameraPos = getThreeCameraPos(camera);
		setFloat("lightPos", threeCameraPos.x, threeCameraPos.y, threeCameraPos.z);
	}

private:
	glm::mat4 createViewTransform(Camera& camera) {
		glm::vec3 threeCameraPos = getThreeCameraPos(camera);
		glm::vec3 cameraFront(
			cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch)),
			sin(glm::radians(camera.pitch)),
			sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch))
		);
		glm::mat4 view = glm::lookAt(threeCameraPos, threeCameraPos + cameraFront, camera.up);
		return view;
	}

	glm::mat4 createProjectionTransform() {
		return glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
	}

	glm::vec3 getThreeCameraPos(Camera& camera) {
		return glm::vec3(camera.pos.x, camera.pos.y, cos(glm::radians(camera.psi)) * camera.pos.z + sin(glm::radians(camera.psi)) * camera.pos.w);
	}
};