#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <sstream>
#include "cuda_runtime.h"

class Vec4 {
public:
	double x;
	double y;
	double z;
	double w;
	__host__ __device__ Vec4(double x, double y, double z, double w) : w(w), x(x), y(y), z(z) {}
	Vec4(glm::vec4 v) : w(v.w), x(v.x), y(v.y), z(v.z) {}
	Vec4(std::vector<double> components) : x(components[0]), y(components[1]),
		z(components[2]), w(components[3]) {}

	__host__ __device__ Vec4 operator+(const Vec4 o) const {
		return Vec4(x + o.x, y + o.y, z + o.z, w + o.w);
	}

	__device__ Vec4 operator-(const Vec4 o) const {
		return Vec4(x - o.x, y - o.y, z - o.z, w - o.w);
	}

	__device__ friend Vec4 operator*(double lhs, const Vec4 rhs) {
		return Vec4(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w);
	}

	Vec4 operator/(double o) const {
		return Vec4(x / o, y / o, z / o, w / o);
	}

	operator std::string() {
		std::stringstream res;
		res << x << ' ' << y << ' ' << z << ' ' << w;
		return res.str();
	}

	__host__ __device__ double dot(const Vec4 o) const {
		return x * o.x + y * o.y + z * o.z + w * o.w;
	}

	__host__ __device__ double magnitude() const {
		return sqrt(x * x + y * y + z * z + w * w);
	}

	Vec4 normalize() const {
		return (*this) / magnitude();
	}

	__device__ static double angle(Vec4 a, Vec4 b) {
		return acos(a.dot(b) / (a.magnitude() * b.magnitude()));
	}
};

