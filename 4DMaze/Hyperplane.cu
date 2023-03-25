#pragma once
#include "Vec4.cu"

class Hyperplane {
public:
	Vec4 normal;
	Vec4 point;
	Hyperplane(Vec4 normal, double e) : normal(normal), point(Vec4(0.0f, 0.0f, 0.0f, 0.0f)) {}
	Hyperplane(Vec4 normal, Vec4 point) : normal(normal), point(point) {}
};