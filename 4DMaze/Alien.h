#pragma once
#include "Vec4.cu"
#include "Mesh.h"
#include "ObjLoader.h"
#include "Hyperplane.cu"
#include "TransformedShader.h"

class Alien {
public:
	Vec4 position;
	Mesh mesh;
	TransformedShader* shader;
	Alien(objl::Loader* loader, TransformedShader* shader, Vec4 position) : position(position), loader(loader), shader(shader),
		mesh(Mesh(shader, loader->LoadedVertices.data(), loader->LoadedVertices.size(),
			loader->LoadedIndices.data(), loader->LoadedIndices.size())) {}

	void updateVisibility(Camera& camera) {
		Hyperplane hyperplane = camera.getHyperplane();
		if (distance(position, hyperplane) < 1.0f) {
			mesh.fillEbo(loader->LoadedVertices.data(), loader->LoadedVertices.size(),
				loader->LoadedIndices.data(), loader->LoadedIndices.size());
			shader->translate = position;
			shader->update(camera);
		}
		else {
			mesh.fillEbo(NULL, 0, NULL, 0);
		}
	}
private:
	objl::Loader* loader;

	double distance(Vec4 pos, Hyperplane hyperplane) {
		Vec4 n = hyperplane.normal;
		double e = hyperplane.normal.dot(hyperplane.point);
		return abs(n.dot(pos) - e) / sqrt(n.dot(n));
	}
};