#pragma once
#include <glm/gtc/type_ptr.hpp>
#include "Vec4.cu"
#include "CameraViewShader.h"

class TransformedShader : public CameraViewShader {
public:
	Vec4 translate;
	float scale;
	TransformedShader(const char* vertexPath, const char* fragmentPath, Vec4 translate, float scale) :
		CameraViewShader(vertexPath, fragmentPath), translate(translate), scale(scale) {}

	void update(Camera& camera) {
		CameraViewShader::update(camera);
		setMatrix("transform", createTranslationTransfrom(camera, translate) * createScaleTransform(scale));
	}

private:
	glm::mat4 createTranslationTransfrom(Camera& camera, Vec4 translate) {
		glm::vec3 threePos(translate.x, translate.y, cos(glm::radians(camera.psi)) * translate.z + sin(glm::radians(camera.psi)) * translate.w);
		return glm::translate(glm::mat4(1.0f), threePos);
	}

	glm::mat4 createScaleTransform(float scale) {
		return glm::scale(glm::mat4(1.0f), glm::vec3(scale, scale, scale));
	}
};