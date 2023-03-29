#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Vec4.cu"
#include "Line.cu"
#include "Hyperplane.cu"
#include "Triangle.cu"
#include "Tetrahedron.cu"
#include "Vertex.cu"

#include "Camera.h"
#include <iostream>

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )
inline void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		std::cerr << cudaGetErrorString(result) << '\n';
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

inline __device__ bool intersect(Line line, Hyperplane hyperplane, Vertex& outVertex) {
	Vec4 p_n = hyperplane.normal;
	Vertex l_0 = line.endpoints[0];
	Vec4 l_d = line.getDir();
	
	double numerator = (hyperplane.point - l_0.pos).dot(p_n);
	if (numerator == 0) {
		// The first endpoint lies in the hyperplane
		outVertex = l_0;
		return true;
	}

	double denominator = p_n.dot(l_d);
	if (denominator == 0) {
		// The line is parallel to the hyperplane
		return false;
	}

	double t = (numerator) / denominator;
	if (t < 0 || t > 1) {
		// The *line* intersects, but ouside the line segment
		return false;
	}

	outVertex = line.tToPoint(t);
	return true;
}

inline __device__
void intersect(Triangle* outTris, unsigned int* outTrisSize, Tetrahedron tetrahedron, Hyperplane hyperplane) {
	Vertex intersectionPoints[6];
	unsigned int numIntersectionPoints = 0;
	Line* tetrahedronLines = tetrahedron.lines;
	for (int i = 0; i < Tetrahedron::NUM_LINES; i++) {
		Line line = tetrahedronLines[i];
		Vertex point(Vec4(0.0f, 0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f)); // dummy
		bool intersected = intersect(line, hyperplane, point);
		if (intersected) {
			intersectionPoints[numIntersectionPoints] = point;
			numIntersectionPoints++;
		}
	}

	if (numIntersectionPoints == 3) {
		// The tetrahedron is askew to the hyperplane, so there is only one tri of intersection
		outTris[0] = Triangle(intersectionPoints[0], intersectionPoints[1], intersectionPoints[2]);
		(*outTrisSize) = 1;
	}
	else if (numIntersectionPoints == 4) {
		// The tetrahedron is askew to the hyperplane and is rotated to produce a convex quadrilateral
		
		// To triangulate the quadrilateral, we first pick an arbitrary three points (0,1,2) to form the first triangle
		// Then, we measure the angle of each pair of points with point 3. The pair that forms the largest angle combine
		// with 3 to form the second triangle
		int a, b;
		double theta031 = abs(Vec4::angle((intersectionPoints[0].pos - intersectionPoints[3].pos), intersectionPoints[1].pos - intersectionPoints[3].pos));
		double theta032 = abs(Vec4::angle((intersectionPoints[0].pos - intersectionPoints[3].pos), intersectionPoints[2].pos - intersectionPoints[3].pos));
		double theta132 = abs(Vec4::angle((intersectionPoints[1].pos - intersectionPoints[3].pos), intersectionPoints[2].pos - intersectionPoints[3].pos));
		if (theta031 > theta032 && theta031 > theta132) {
			a = 0;
			b = 1;
		}
		else if (theta032 > theta031 && theta032 > theta132) {
			a = 0;
			b = 2;
		}
		else {
			a = 1;
			b = 2;
		}

		outTris[0] = Triangle(intersectionPoints[0], intersectionPoints[1], intersectionPoints[2]);
		outTris[1] = Triangle(intersectionPoints[a], intersectionPoints[b], intersectionPoints[3]);
		*outTrisSize = 2;
	}
	else if (numIntersectionPoints == 6) {
		// The tetrahedron is entirely embedded in the hyperplane, so return all 4 triangles that make up the tetra
		Triangle* tris = tetrahedron.getTriangles();
		outTris[0] = tris[0];
		outTris[1] = tris[1];
		outTris[2] = tris[2];
		outTris[3] = tris[3];
		*outTrisSize = Tetrahedron::NUM_TRIANGLES;
		free(tris);
	}
	else {
		// The tetrahedron does not intersect, or only intersects at one point, so return no triangles
		*outTrisSize = 0;
	}
}

inline __global__
void getTrianglesKernel(Triangle* outTris, unsigned int* outNumTrisForTetrahedon, Tetrahedron* tetrahedra, unsigned int numTetrahedra, Hyperplane hyperplane) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > numTetrahedra - 1) {
		return;
	}
	Tetrahedron tetrahedron = tetrahedra[i];
	Triangle tris[Tetrahedron::NUM_TRIANGLES];
	unsigned int numTris = 0;
	intersect(tris, &numTris, tetrahedron, hyperplane);

	outNumTrisForTetrahedon[i] = numTris;
	for (unsigned int j = 0; j < numTris; j++) {
		outTris[i * Tetrahedron::NUM_TRIANGLES + j] = tris[j];
	}
}

inline std::vector<Triangle> getTriangles(const std::vector<Tetrahedron>& tetrahedra, Hyperplane hyperplane) {

	Triangle* devTriangles = 0;
	checkCudaErrors(cudaMalloc((void**)&devTriangles, sizeof(Triangle) * Tetrahedron::NUM_TRIANGLES * tetrahedra.size()));

	Tetrahedron* devTetrahedra = 0;
	checkCudaErrors(cudaMalloc((void**)&devTetrahedra, sizeof(Tetrahedron) * tetrahedra.size()));
	cudaMemcpy(devTetrahedra, tetrahedra.data(), sizeof(Tetrahedron) * tetrahedra.size(), cudaMemcpyHostToDevice);

	unsigned int* devNumTriangles;
	checkCudaErrors(cudaMalloc((void**)&devNumTriangles, sizeof(unsigned int) * tetrahedra.size()));

	const int MAX_THREADS_PER_BLOCK = 512;
	int numBlocks = (tetrahedra.size() / MAX_THREADS_PER_BLOCK) + 1;
	getTrianglesKernel<<<numBlocks, MAX_THREADS_PER_BLOCK >>>(devTriangles, devNumTriangles, devTetrahedra, tetrahedra.size(), hyperplane);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	unsigned int* trianglesSize = (unsigned int*)malloc(sizeof(unsigned int) * tetrahedra.size());
	checkCudaErrors(cudaMemcpy(trianglesSize, devNumTriangles, sizeof(unsigned int) * tetrahedra.size(), cudaMemcpyDeviceToHost));
	Triangle* triangles = (Triangle*)malloc(sizeof(Triangle) * Tetrahedron::NUM_TRIANGLES * tetrahedra.size());
	checkCudaErrors(cudaMemcpy(triangles, devTriangles, sizeof(Triangle) * Tetrahedron::NUM_TRIANGLES * tetrahedra.size(), cudaMemcpyDeviceToHost));

	std::vector<Triangle> triangleVec;
	for (int i = 0; i < tetrahedra.size(); i++) {
		unsigned int numTrisForTetrahedron = trianglesSize[i];
		for (int j = 0; j < numTrisForTetrahedron; j++) {
			triangleVec.push_back(triangles[i * Tetrahedron::NUM_TRIANGLES + j]);
		}
	}

	cudaFree(devTriangles);
	cudaFree(devNumTriangles);
	cudaFree(devTetrahedra);
	free(trianglesSize);
	free(triangles);

	return triangleVec;
}


inline std::vector<float> getVertices(std::vector<Tetrahedron> tetrahedra, Camera& camera) {
	Hyperplane hyperplane = camera.getHyperplane();
	std::vector<Triangle> triangles = getTriangles(tetrahedra, hyperplane);
	std::vector<float> vertices;
	for (int i = 0; i < triangles.size(); i++) {
		Triangle triangle = triangles[i];
		Vec3 normal = triangle.getNormal(camera.psi);
		for (int j = 0; j < sizeof(triangle.points) / sizeof(Vertex); j++) {
			Vertex point = triangle.points[j];
			vertices.push_back(point.pos.x);
			vertices.push_back(point.pos.y);
			vertices.push_back(point.pos.z);
			vertices.push_back(point.pos.w); // In OpenGL, w is the fourth comp of vec4
			vertices.push_back(point.texCoord.u);
			vertices.push_back(point.texCoord.v);
			vertices.push_back(point.texCoord.t);
			vertices.push_back(normal.u);
			vertices.push_back(normal.v);
			vertices.push_back(normal.t); // TODO: rename these
		}
	}
	return vertices;
}