#pragma once
#include "Vec4.cu"
#include <vector>
#include "Vertex.cu"

class Triangle {
public:
	Vertex points[3];
	__device__ Triangle(Vertex p1, Vertex p2, Vertex p3) : points{ p1, p2, p3 } {}
	Triangle(std::vector<Vertex> points) : points{ points[0], points[1], points[2] } {}
	__device__ Triangle() : points{ Vertex(), Vertex(), Vertex() } {}

	Vec3 getNormal(float psi) {
		Vec3 a = threePos(points[0].pos, psi);
		Vec3 b = threePos(points[1].pos, psi);
		Vec3 c = threePos(points[2].pos, psi);

		return (b - a).cross(c - a).normalize();
	}

private:
	Vec3 threePos(Vec4 pos, float psi) {
		return Vec3(pos.x, pos.y, cos(glm::radians(psi)) * pos.z + sin(glm::radians(psi)) * pos.w);
	}
};