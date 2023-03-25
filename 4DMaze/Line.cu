#pragma once
#include "Vec4.cu"
#include "Vertex.cu"

class Line {
public:
	Vertex endpoints[2];
	__device__ Line(Vertex p1, Vertex p2) : endpoints{ p1, p2 } {}

	__device__ Vec4 getDir() const {
		return endpoints[1].pos - endpoints[0].pos;
	}

	__device__ Vertex tToPoint(double t) const {
		Vec4 pos = endpoints[0].pos + t * getDir();
		Vec3 texCoord = t * endpoints[1].texCoord + (1.0 - t) * endpoints[0].texCoord;
		return Vertex(pos, texCoord);
	}
};