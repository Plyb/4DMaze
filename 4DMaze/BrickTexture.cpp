#pragma once
#include <cstdlib>

class BrickTexture {
public:
	const unsigned int brickHeight = 10;
	const unsigned int numBricks = 10;
	const unsigned int sideLength = brickHeight * numBricks;
	unsigned char* pixels;

	BrickTexture() {
		pixels = new unsigned char[sideLength * sideLength * sideLength * 4];
		for (int x = 0; x < numBricks / 2; x++) {
			for (int y = 0; y < numBricks; y++) {
				for (int z = 0; z < numBricks / 2; z++) {
					unsigned char redness = clamp(rand() % 256, 20, 235);
					for (int u = x * brickHeight * 2; u < (x + 1) * brickHeight * 2; u++) {
						for (int v = y * brickHeight; v < (y + 1) * brickHeight; v++) {
							for (int t = z * brickHeight * 2; t < (z + 1) * brickHeight * 2; t++) {
								int i = (u * sideLength * sideLength + v * sideLength + t) * 4;

								if (u == x * brickHeight * 2 || u == (x + 1) * brickHeight * 2
									|| v == y * brickHeight || v == v < (y + 1) * brickHeight
									|| t == z * brickHeight * 2 || t == t < (z + 1) * brickHeight * 2) {
									pixels[i + 0] = 0;
									pixels[i + 1] = 0;
									pixels[i + 2] = 0;
									pixels[i + 3] = 255;
								}
								else {
									unsigned char noise = rand() % 40 - 20;
									pixels[i + 0] = redness + noise;
									pixels[i + 1] = 0;
									pixels[i + 2] = 0;
									pixels[i + 3] = 255;
								}
							}
						}
					}
				}
			}
		}
	}

	~BrickTexture() {
		delete pixels;
	}
private:
	unsigned char clamp(unsigned char val, unsigned char min, unsigned char max) {
		if (val < min) {
			return min;
		}
		if (val > max) {
			return max;
		}
		return val;
	}
};