#pragma once
#include <glm/glm.hpp>
#include <functional>
#include "Hyperplane.cu"
#include "Scene.h"

class Camera {
public:
	const float CAMERA_SPEED = 2.5f;

	Scene& scene;

	Camera(Scene& scene) : scene(scene) {}

	glm::vec4 pos = glm::vec4(0.5f, 0.0f, 0.5f, 0.5f);
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

	float yaw = 0.0f;
	float pitch = 0.0f;
	float psi = 0.0f;

	void onMousePosChange(double xoffset, double yoffset);

	bool update(float deltaTime, std::function<bool(int)> keyPressed);

	Hyperplane getHyperplane() {
		return Hyperplane(Vec4(0.0f, 0.0f, sin(glm::radians(psi)), -cos(glm::radians(psi))), Vec4(pos));
	}
private:
	glm::vec4 front = computeFront();
	glm::vec4 side = computeSide();

	glm::vec4 computeFront();
	glm::vec4 computeSide();

	glm::vec4 isFreeSpace(glm::vec4 pos, glm::vec4 stepDir);
	glm::vec4 intersectsTesseract(glm::vec4 pos, Tesseract& tesseract, glm::vec4 stepDir);
	bool insersectsAlien(glm::vec4 pos);
};