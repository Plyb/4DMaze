#pragma once
#include "Camera.h"
#include "Alien.h"
#include <GLFW/glfw3.h>
#include <iostream>

bool Camera::update(float deltaTime, std::function<bool(int)> keyPressed) {
	float cameraSpeed = deltaTime * CAMERA_SPEED;
	const float ROT_SPEED = 3.0f;

	glm::vec4 newPos = pos;
	if (keyPressed(GLFW_KEY_W)) {
		newPos += cameraSpeed * front;
	}
	if (keyPressed(GLFW_KEY_S)) {
		newPos -= cameraSpeed * front;
	}
	if (keyPressed(GLFW_KEY_A)) {
		newPos -= side * cameraSpeed;
	}
	if (keyPressed(GLFW_KEY_D)) {
		newPos += side * cameraSpeed;
	}

	if (isFreeSpace(newPos) || true) {
		pos = newPos;
	}
	if (insersectsAlien(newPos)) {
		pos = glm::vec4(0.5f, 0.5f, 0.5f, 0.5f);
		double alienCoord = scene.mazeSize * 2 - 1.5;
		scene.alien.position = Vec4(alienCoord, alienCoord, alienCoord, alienCoord);
		scene.loadMaze();
		return true;
	}


	if (keyPressed(GLFW_KEY_Q)) {
		psi += ROT_SPEED * cameraSpeed;
		front = computeFront();
		side = computeSide();
		return true;
	}
	if (keyPressed(GLFW_KEY_E)) {
		psi -= ROT_SPEED * cameraSpeed;
		front = computeFront();
		side = computeSide();
		return true;
	}
	std::cout << psi << '\n';
	std::cout << pos.x << ' ' << pos.y << ' ' << pos.z << ' ' << pos.w << '\n';
	return false;
}

void Camera::onMousePosChange(double xoffset, double yoffset) {
	yaw += xoffset;
	pitch += yoffset;

	if (pitch > 89.0f) {
		pitch = 89.0f;
	}
	else if (pitch < -89.0f) {
		pitch = -89.0f;
	}
	front = computeFront();
	side = computeSide();
}

glm::vec4 Camera::computeFront() {
	return glm::vec4(
		cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
		sin(glm::radians(pitch)),
		sin(glm::radians(yaw)) * cos(glm::radians(pitch)) * cos(glm::radians(psi)),
		sin(glm::radians(yaw)) * cos(glm::radians(pitch)) * sin(glm::radians(psi))
	);
}

glm::vec4 Camera::computeSide() {
	return glm::normalize(glm::vec4(
		-sin(glm::radians(yaw)) * cos(glm::radians(pitch)),
		0.0f,
		cos(glm::radians(yaw)) * cos(glm::radians(pitch)) * cos(glm::radians(psi)),
		cos(glm::radians(yaw)) * cos(glm::radians(pitch)) * sin(glm::radians(psi))
	));
}

bool Camera::isFreeSpace(glm::vec4 pos) {
	for (int i = 0; i < scene.tesseracts.size(); i++) {
		if (intersectsTesseract(pos, scene.tesseracts[i])) {
			return false;
		}
	}
	return true;
}

bool Camera::intersectsTesseract(glm::vec4 pos, Tesseract& tesseract) {
	Vec4 tessCenter = tesseract.lowestCorner + Vec4(0.5f, 0.5f, 0.5f, 0.5f);
	return abs(pos.x - tessCenter.x) < 1.0f
		&& abs(pos.y - tessCenter.y) < 1.0f
		&& abs(pos.z - tessCenter.z) < 1.0f
		&& abs(pos.w - tessCenter.w) < 1.0f;
}

bool Camera::insersectsAlien(glm::vec4 pos) {
	Vec4 alienPos = scene.alien.position;
	return abs(glm::length(pos - glm::vec4(alienPos.x, alienPos.y, alienPos.z, alienPos.w))) < 1.0f;
	return true;
}