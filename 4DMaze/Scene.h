#pragma once
#include <vector>
#include "Tesseract.cpp"
#include "Maze.h"

class Alien;
class Scene {
public:
	std::vector<Tesseract> tesseracts;
	std::vector<Tetrahedron> tetrahedra;
	Alien& alien;
	int mazeSize = 3;

	Scene(Alien& alien) : alien(alien) {
		for (int i = -3; i < 3; i++) {
			for (int j = -3; j < 3; j++) {
				tesseracts.push_back(Tesseract(Vec4(-0.5, i, j, -0.5)));
			}
		}
		tetrahedra = getTetrahedra();
	}

	bool shouldFillCell(Maze& maze, int x, int y, int z, int w, int levelWidth) {
		if (x == -1 || x == levelWidth - 1 || y == -1 || y == levelWidth - 1
			|| z == -1 || z == levelWidth - 1 || w == -1 || w == levelWidth - 1) {
			return true;
		}
		else if ((x % 2) + (y % 2) + (z % 2) + (w % 2) > 1) {
			return true;
		}
		else if (x % 2 || y % 2 || z % 2 || w % 2) {
			MazeCell cell = maze.cells[x / 2 + x % 2][y / 2 + y % 2][z / 2 + z % 2][w / 2 + w % 2];
			if (x % 2) {
				if (cell.left) {
					return true;
				}
			}
			else if (y % 2) {
				if (cell.down) {
					return true;
				}
			}
			else if (z % 2) {
				if (cell.back) {
					return true;
				}
			}
			else if (w % 2) {
				if (cell.wDown) {
					return true;
				}
			}
			return false;
		}
	}

	void loadMaze() {
		Maze maze(mazeSize);
		tesseracts.clear();

		int levelWidth = maze.size * 2;
		for (int x = -1; x < levelWidth; x++) {
			for (int y = -1; y < levelWidth; y++) {
				for (int z = -1; z < levelWidth; z++) {
					for (int w = -1; w < levelWidth; w++) {
						if (shouldFillCell(maze, x, y, z, w, levelWidth)) {
							tesseracts.push_back(Tesseract(Vec4(x, y, z, w)));
						}
						
					}
				}
			}
		}
		tetrahedra = getTetrahedra();
		mazeSize++;
	}

private:
	std::vector<Tetrahedron> getTetrahedra() {
		std::vector<Tetrahedron> allTetras;
		for (int i = 0; i < tesseracts.size(); i++) {
			std::vector<Tetrahedron> tetras = tesseracts.at(i).getTetrahedra();
			allTetras.insert(allTetras.end(), tetras.begin(), tetras.end());
		}
		return allTetras;
	}
};