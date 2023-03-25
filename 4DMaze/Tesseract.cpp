#pragma once
#include "Vec4.cu"
#include "Cube.cpp"

class Tesseract {
public:
	Vec4 lowestCorner;
	Tesseract(Vec4 lowestCorner) : lowestCorner(lowestCorner) {}

	std::vector<Cube> getCubes() {
		return {
			Cube(lowestCorner, 0),
			Cube(lowestCorner, 1),
			Cube(lowestCorner, 2),
			Cube(lowestCorner, 3),
			Cube(lowestCorner + Vec4(1, 0, 0, 0), 0),
			Cube(lowestCorner + Vec4(0, 1, 0, 0), 1),
			Cube(lowestCorner + Vec4(0, 0, 1, 0), 2),
			Cube(lowestCorner + Vec4(0, 0, 0, 1), 3),
		};
	}

	std::vector<Tetrahedron> getTetrahedra() {
		std::vector<Tetrahedron> allTetras;
		std::vector<Cube> cubes = getCubes();
		for (int i = 0; i < cubes.size(); i++) {
			std::vector<Tetrahedron> tetras = cubes[i].getTetrahedra();
			allTetras.insert(allTetras.end(), tetras.begin(), tetras.end());
		}
		return allTetras;
	}
};