#pragma once
#include "Shader.h"
#include <glad/glad.h>
#include "ObjLoader.h"

class Mesh {
public:
	Shader* shader;
	unsigned int numElemsInVertexBuffer;

	bool usesElementArray;
	unsigned int numIndices;

	Mesh(Shader* shader, float* vertexBuffer, unsigned int bufferSize) :
		shader(shader), numElemsInVertexBuffer(bufferSize), usesElementArray(false), numIndices(0), ebo(0)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		fillVbo(vertexBuffer, bufferSize);

		// Position
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, vertexSize * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		// uvt coords
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertexSize * sizeof(float), (void*)(4 * sizeof(float)));
		glEnableVertexAttribArray(1);

		// Normal
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_TRUE, vertexSize * sizeof(float), (void*)(7 * sizeof(float)));
		glEnableVertexAttribArray(2);

		glBindVertexArray(0);
	}

	Mesh(Shader* shader, objl::Vertex* vertexBuffer, unsigned int numElemsInVertexBuffer, unsigned int* indices, unsigned int numIndices) :
		shader(shader), numElemsInVertexBuffer(numElemsInVertexBuffer), usesElementArray(true),
		numIndices(numIndices)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, numElemsInVertexBuffer * sizeof(objl::Vertex), vertexBuffer, GL_STATIC_DRAW);

		glGenBuffers(1, &ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(unsigned int), indices, GL_STATIC_DRAW);

		// Position
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(objl::Vertex), (void*)offsetof(objl::Vertex, Position));
		glEnableVertexAttribArray(0);

		// Normal
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, sizeof(objl::Vertex), (void*)offsetof(objl::Vertex, Normal));
		glEnableVertexAttribArray(1);

		glBindVertexArray(0);
	}

	void fillVbo(float* vertexBuffer, unsigned int numElemsInVertexBuffer) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, numElemsInVertexBuffer * sizeof(float), vertexBuffer, GL_STATIC_DRAW);
		this->numElemsInVertexBuffer = numElemsInVertexBuffer;
	}

	void fillEbo(objl::Vertex* vertexBuffer, unsigned int numElemsInVertexBuffer, unsigned int* indices, unsigned int numIndices) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, numElemsInVertexBuffer * sizeof(objl::Vertex), vertexBuffer, GL_STATIC_DRAW);
		this->numElemsInVertexBuffer = numElemsInVertexBuffer;

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(unsigned int), indices, GL_STATIC_DRAW);
		this->numIndices = numIndices;
	}

	void draw(Camera& camera) {
		shader->use();
		shader->update(camera);
		if (usesElementArray) {
			drawEbo();
		}
		else {
			drawVao();
		}
	}
private:
	const unsigned int vertexSize = 10;
	unsigned int vao, vbo, ebo;

	void drawVao() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, numElemsInVertexBuffer / vertexSize);
	}

	void drawEbo() {
		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
	}
};