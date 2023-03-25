#pragma once
#include "MazeCell.h"
#include <stack>
#include <set>
#include <vector>

class Maze {
public:
	std::vector<std::vector<std::vector<std::vector<MazeCell>>>> cells;
	unsigned int size;
	unsigned int x = 0, y = 0, z = 0, w = 0;

	Maze(unsigned int size) : size(size) {
		generate(size);
	}

	void generate(int size) {
		cells.clear();
		this->size = size;
		for (int x = 0; x < size; x++) {
			cells.push_back(std::vector<std::vector<std::vector<MazeCell>>>());
			visitedCells.push_back(std::vector<std::vector<std::vector<bool>>>());
			for (int y = 0; y < size; y++) {
				cells[x].push_back(std::vector<std::vector<MazeCell>>());
				visitedCells[x].push_back(std::vector<std::vector<bool>>());
				for (int z = 0; z < size; z++) {
					cells[x][y].push_back(std::vector<MazeCell>());
					visitedCells[x][y].push_back(std::vector<bool>());
					for (int w = 0; w < size; w++) {
						cells[x][y][z].push_back(MazeCell(x, y, z, w));
						visitedCells[x][y][z].push_back(false);
					}
				}
			}
		}

		while (true) {
			while (isBoxedIn()) {
				visitedCells[x][y][z][w] = true;
				MazeCell previousCell = backtrackPath.top();
				backtrackPath.pop();
				if (previousCell.x == 0 && previousCell.y == 0
					&& previousCell.z == 0 && previousCell.w == 0) {
					return;
				}
				x = previousCell.x;
				y = previousCell.y;
				z = previousCell.z;
				w = previousCell.w;
			}
			DIR dir = getStepDir();
			stepInDir(dir);
		}
	}

private:
	std::stack<MazeCell> backtrackPath;
	std::vector<std::vector<std::vector<std::vector<bool>>>> visitedCells;
	enum DIR {
		LEFT, RIGHT, DOWN, UP, BACK, FORWARD, WDOWN, WUP,
	};

	bool isBoxedIn() {
		return visited(x - 1, y, z, w) && visited(x + 1, y, z, w)
			&& visited(x, y - 1, z, w) && visited(x, y + 1, z, w)
			&& visited(x, y, z - 1, w) && visited(x, y, z + 1, w)
			&& visited(x, y, z, w - 1) && visited(x, y, z, w + 1);
	}

	bool visited(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {
		if (x < 0 || x > size - 1 || y < 0 || y > size - 1
			|| z < 0 || z > size - 1 || w < 0 || w > size - 1) {
			return true;
		}
		return visitedCells[x][y][z][w];
	}

	void stepInDir(DIR dir) {
		backtrackPath.push(cells[x][y][z][w]);
		visitedCells[x][y][z][w] = true;
		switch (dir) {
			case LEFT:
				cells[x][y][z][w].left = false;
				x -= 1;
				break;
			case RIGHT:
				x += 1;
				cells[x][y][z][w].left = false;
				break;
			case DOWN:
				cells[x][y][z][w].down = false;
				y -= 1;
				break;
			case UP:
				y += 1;
				cells[x][y][z][w].down = false;
				break;
			case BACK:
				cells[x][y][z][w].back = false;
				z -= 1;
				break;
			case FORWARD:
				z += 1;
				cells[x][y][z][w].back = false;
				break;
			case WDOWN:
				cells[x][y][z][w].wDown = false;
				w -= 1;
				break;
			case WUP:
				w += 1;
				cells[x][y][z][w].wDown = false;
				break;
		}
	}

	DIR getStepDir() {
		while (true) {
			unsigned int dir = rand() % 8;
			if (   (dir == RIGHT  && !(x == size - 1  || visited(x + 1, y, z, w)))
				|| (dir == UP      && !(y == size - 1 || visited(x, y + 1, z, w)))
				|| (dir == FORWARD && !(z == size - 1 || visited(x, y, z + 1, w)))
				|| (dir == WUP     && !(w == size - 1 || visited(x, y, z, w + 1)))
				|| (dir == LEFT    && !(x == 0        || visited(x - 1, y, z, w)))
				|| (dir == DOWN    && !(y == 0        || visited(x, y - 1, z, w)))
				|| (dir == BACK    && !(z == 0        || visited(x, y, z - 1, w)))
				|| (dir == WDOWN   && !(w == 0        || visited(x, y, z, w - 1)))) {
				return (DIR) dir;
			}
		}
	}
};