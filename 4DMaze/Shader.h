#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <glm/glm.hpp>
#include "Camera.h"

class Shader {
public:
	unsigned int ID;

	Shader(const char* vertexPath, const char* fragmentPath);

	// activates the shader
	void use();

	virtual void update(Camera& camera);

	// uniform setters
	void setBool(const std::string& name, bool value) const;
	void setInt(const std::string& name, int value) const;
	void setFloat(const std::string& name, float value) const;
	void setFloat(const std::string& name, float x, float y, float z) const;
	void setFloat(const std::string& name, float w, float x, float y, float z) const;
	void setMatrix(const std::string& name, const glm::mat4& matrix);
};

#endif