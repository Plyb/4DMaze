#pragma once
#include "Vec4.cu"
#include <vector>
#include "Line.cu"
#include "Triangle.cu"

class Tetrahedron {
public:
	Vertex points[4];
	__host__ __device__ Tetrahedron(Vertex p1, Vertex p2, Vertex p3, Vertex p4) : points{ p1, p2, p3, p4 } {}

	const static unsigned int NUM_LINES = 6;
	Line lines[NUM_LINES] = {
		Line(points[0], points[1]),
		Line(points[1], points[2]),
		Line(points[2], points[3]),
		Line(points[3], points[0]),
		Line(points[0], points[2]),
		Line(points[1], points[3])
	};

	const static unsigned int NUM_TRIANGLES = 4;
	__device__ Triangle* getTriangles() const {
		Triangle* triangles = (Triangle*)malloc(sizeof(Triangle) * NUM_TRIANGLES);
		triangles[0] = Triangle(points[0], points[1], points[2]);
		triangles[1] = Triangle(points[0], points[3], points[1]);
		triangles[2] = Triangle(points[1], points[3], points[2]);
		triangles[3] = Triangle(points[0], points[2], points[3]);
		return triangles;
	}
};