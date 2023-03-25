#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "BrickShader.cpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include "Alien.h"
#include "Slice.cu"

float deltaTime = 0.0f;
float lastFrame = 0.0f;

float lastX = 400, lastY = 300;
bool firstMouse = true;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

std::vector<float> buildBrickVertexBuffer(Camera& camera) {
	std::vector<Tetrahedron> tetras = camera.scene.tetrahedra;
	return getVertices(tetras, camera);
}

std::function<void(GLFWwindow* window, double xpos, double)> mouseLambda;
void onMouseUpdate(GLFWwindow* window, double xpos, double ypos, Camera& camera) {
	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;

	const float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	camera.onMousePosChange(xoffset, yoffset);
}
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	mouseLambda(window, xpos, ypos);
}

void processInput(GLFWwindow* window, Mesh* brickMesh, Camera& camera) {
	auto keyPressed = [&window](int key) {
		return glfwGetKey(window, key) == GLFW_PRESS;
	};

	if (keyPressed(GLFW_KEY_ESCAPE)) {
		glfwSetWindowShouldClose(window, true);
	}

	bool regenerateScene = camera.update(deltaTime, keyPressed);

	if (regenerateScene) {
		std::vector<float> brickVerts = buildBrickVertexBuffer(camera);
		brickMesh->fillVbo(brickVerts.data(), brickVerts.size());

		camera.scene.alien.updateVisibility(camera);
	}
}

int main() {
	int startSize = 3;
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(800, 600, "4D Maze", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
			return -1;
	}


	objl::Loader loader;
	loader.LoadFile("./alien.obj");
	double alienCoord = startSize * 2 - 1.5;
	Vec4 alienPos(alienCoord, alienCoord, alienCoord, alienCoord);
	TransformedShader objShader("./objShader.vert", "./objShader.frag", alienPos, 0.05f);
	Alien alien(&loader, &objShader, alienPos);

	Scene scene(alien);
	scene.loadMaze();
	Camera camera(scene);
	alien.updateVisibility(camera);

	BrickShader brickShader;

	std::vector<float> brickVertices = buildBrickVertexBuffer(camera);
	Mesh brickMesh(&brickShader, brickVertices.data(), brickVertices.size());

	mouseLambda = [&camera](GLFWwindow* window, double xpos, double ypos) {
		onMouseUpdate(window, xpos, ypos, camera);
	};

	glViewport(0, 0, 800, 600);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glEnable(GL_DEPTH_TEST);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	while (!glfwWindowShouldClose(window)) {
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window, &brickMesh, camera);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		brickMesh.draw(camera);
		alien.mesh.draw(camera);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	

	glfwTerminate();
	return 0;
}