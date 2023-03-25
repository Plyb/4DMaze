#pragma once
#include "Vec4.cu"
#include "Vec3.cu"

class Vertex {
public:
	Vec4 pos;
	Vec3 texCoord;
	__host__ __device__ Vertex(Vec4 pos, Vec3 texCoord) : pos(pos), texCoord(texCoord) {}
	__device__ Vertex() : pos(Vec4(0, 0, 0, 0)), texCoord(Vec3(0, 0, 0)) {}
};