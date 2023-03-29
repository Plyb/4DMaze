#pragma once
#include "Camera.h"
#include "Alien.h"
#include <GLFW/glfw3.h>
#include <iostream>

bool Camera::update(float deltaTime, std::function<bool(int)> keyPressed) {
	float cameraSpeed = deltaTime * CAMERA_SPEED;
	const float ROT_SPEED = 3.0f;

	glm::vec4 stepDir = glm::vec4(0, 0, 0, 0);
	if (keyPressed(GLFW_KEY_W)) {
		stepDir += cameraSpeed * front;
	}
	if (keyPressed(GLFW_KEY_S)) {
		stepDir -= cameraSpeed * front;
	}
	if (keyPressed(GLFW_KEY_A)) {
		stepDir -= side * cameraSpeed;
	}
	if (keyPressed(GLFW_KEY_D)) {
		stepDir += side * cameraSpeed;
	}
	glm::vec4 newPos = pos + stepDir;

	pos = isFreeSpace(newPos, stepDir);
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

glm::vec4 Camera::isFreeSpace(glm::vec4 pos, glm::vec4 stepDir) {
	glm::vec4 newPos = pos;
	for (int i = 0; i < scene.tesseracts.size(); i++) {
		glm::vec4 translation = intersectsTesseract(pos, scene.tesseracts[i], stepDir);
		if (translation != glm::vec4(0, 0, 0, 0)) {
			newPos += translation;
		}
	}
	return newPos;
}

glm::vec4 Camera::intersectsTesseract(glm::vec4 pos, Tesseract& tesseract, glm::vec4 stepDir) {
	glm::vec4 translation = glm::vec4(0, 0, 0, 0);
	Vec4 tessCenter = tesseract.lowestCorner + Vec4(0.5f, 0.5f, 0.5f, 0.5f);
	const float allowedDistance = 0.75f;

	double adx = abs(pos.x - tessCenter.x);
	double ady = abs(pos.y - tessCenter.y);
	double adz = abs(pos.z - tessCenter.z);
	double adw = abs(pos.w - tessCenter.w);
	if (adx < allowedDistance && ady < allowedDistance
		&& adz < allowedDistance && adw < allowedDistance
	) {
		if (adx >= ady && adx >= adz && adx >= adw) {
			double sign = (pos.x - tessCenter.x) / adx;
			translation.x = adx == 0 ? 0 : (sign * allowedDistance) - (pos.x - tessCenter.x);
		}
		if (ady >= adx && ady >= adz && ady >= adw) {
			double sign = (pos.y - tessCenter.y) / ady;
			translation.y = ady == 0 ? 0 : (sign * allowedDistance) - (pos.y - tessCenter.y);
		}
		if ((adz >= adx && adz >= ady && adz >= adw) || (adw >= adx && adw >= ady && adw >= adz)) {
			translation.z = -CAMERA_SPEED * stepDir.z;
			translation.w = -CAMERA_SPEED * stepDir.w;
		}
	}
	return translation;
}

bool Camera::insersectsAlien(glm::vec4 pos) {
	Vec4 alienPos = scene.alien.position;
	return abs(glm::length(pos - glm::vec4(alienPos.x, alienPos.y, alienPos.z, alienPos.w))) < 1.0f;
	return true;
}