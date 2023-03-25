#pragma once

class MazeCell {
public:
	bool left, down, back, wDown;
	unsigned int x, y, z, w;
	MazeCell(unsigned int x, unsigned int y, unsigned int z, unsigned int w) :
		x(x), y(y), z(z), w(w), left(true), down(true), back(true), wDown(true) {}
};