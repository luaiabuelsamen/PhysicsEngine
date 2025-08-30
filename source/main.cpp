#include "MultiMechanicalSystem.h"
#include <GLFW/glfw3.h>
// Forward declarations for mouse event handlers (GLFWwindow is now defined)
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);

// Forward declaration for screenToWorldX
float screenToWorldX(double screenX, int windowWidth);
#include <GL/glut.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <Eigen/Dense>

#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Multi-DOF 1D spring-mass system parameters

constexpr int numMasses = 3;
std::vector<float> masses(numMasses, 1.0f);
std::vector<float> dampings(numMasses, 0.1f);
std::vector<float> springConstants(numMasses, 1.0f);
std::vector<float> initialPositions = {-0.9f, -0.3f, 0.4f};
std::vector<float> initialVelocities(numMasses, 0.0f);
// Couplings: connect mass 0-1, 1-2
std::vector<std::pair<int, int>> couplings = { {0,1}, {1,2} };
std::vector<float> couplingConstants = {1.0f, 1.0f};
// Wall spring constant (same as others for now)
float wallSpringConstant = 1.0f;
float wallPosition = -1.0f;

MultiMechanicalSystem multiSystem(masses, dampings, springConstants, initialPositions, initialVelocities, couplings, couplingConstants);
std::vector<float> positions = initialPositions;
std::vector<float> velocities = initialVelocities;
float simTime = 0.0f;
bool oscillationStarted = true;
int draggedBlock = -1; // index of block being dragged, -1 if none
float dragOffset = 0.0f;
double lastMouseX = 0.0;
// Mouse event handlers for dragging blocks

// --- Place these at the end of the file, after main() ---

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    float mouseWorldX = screenToWorldX(mouseX, width);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            // Check if mouse is over any block
            for (int i = 0; i < numMasses; ++i) {
                if (std::abs(mouseWorldX - positions[i]) < 0.07f) {
                    draggedBlock = i;
                    dragOffset = positions[i] - mouseWorldX;
                    lastMouseX = mouseWorldX;
                    break;
                }
            }
        } else if (action == GLFW_RELEASE && draggedBlock != -1) {
            // Set velocity based on drag
            velocities[draggedBlock] = (mouseWorldX - lastMouseX) / 0.016f;
            draggedBlock = -1;
        }
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (draggedBlock != -1) {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        float mouseWorldX = screenToWorldX(xpos, width);
        positions[draggedBlock] = mouseWorldX + dragOffset;
        lastMouseX = mouseWorldX;
    }
}

// Function declarations
float screenToWorldX(double screenX, int windowWidth);
bool isMouseOverCube(float mouseWorldX, float cubeWorldX);
bool isMouseOverCube(float mouseWorldX, const Eigen::VectorXf& cubeWorldX);
void drawMass(float xPos, float yPos = 0.0f, bool isHovered = false);
void drawSpring(float x1, float x2, float y = 0.0f);
void drawGrid(float spacing, int numLines);
void drawText(const std::string& text, float x, float y, float scale = 1.0f);


// Function implementations
float screenToWorldX(double screenX, int windowWidth) {
    // Convert screen coordinates to world coordinates
    // Assuming world coordinates go from -2.0 to 2.0
    return (float)(screenX / windowWidth * 4.0 - 2.0);
}

bool isMouseOverCube(float mouseWorldX, float cubeWorldX) {
    float size = 0.05f; // Same size as in drawMass
    return (mouseWorldX >= cubeWorldX - size && mouseWorldX <= cubeWorldX + size);
}

bool isMouseOverCube(float mouseWorldX, const Eigen::VectorXf& cubeWorldX) {
    return isMouseOverCube(mouseWorldX, cubeWorldX(0));
}

void drawMass(float xPos, float yPos, bool isHovered) {
    glBegin(GL_QUADS);
    float size = 0.05f;
    if (isHovered) {
        glColor4f(1.0f, 0.5f, 0.3f, 1.0f);
    } else {
        glColor4f(0.9f, 0.3f, 0.2f, 1.0f);
    }
    glVertex2f(xPos - size, yPos - size);
    glVertex2f(xPos + size, yPos - size);
    glVertex2f(xPos + size, yPos + size);
    glVertex2f(xPos - size, yPos + size);
    glEnd();
}

void drawSpring(float x1, float x2, float y) {
    glBegin(GL_LINE_STRIP);
    glColor4f(0.4f, 0.8f, 1.0f, 1.0f);
    glLineWidth(2.0f);
    int segments = 30;
    float amplitude = 0.08f;
    for (int i = 0; i <= segments; i++) {
        float t = (float)i / segments;
        float x = x1 + (x2 - x1) * t;
        float yOffset = amplitude * sin(t * 10 * M_PI) * (1.0f - 0.5f * std::abs(t - 0.5f));
        glVertex2f(x, y + yOffset);
    }
    glEnd();
}

void drawGrid(float spacing, int numLines) {
    glColor3f(0.2f, 0.2f, 0.25f); // Darker grid for better contrast
    glLineWidth(1.0f);
    glBegin(GL_LINES);

    // Vertical lines
    for (int i = -numLines; i <= numLines; ++i) {
        float x = i * spacing;
        glVertex2f(x, -1.5f);
        glVertex2f(x, 1.5f);
    }

    // Horizontal lines
    for (int i = -numLines; i <= numLines; ++i) {
        float y = i * spacing;
        glVertex2f(-2.0f, y);
        glVertex2f(2.0f, y);
    }

    glEnd();
}

void drawText(const std::string& text, float x, float y, float scale) {
    glPushMatrix();
    glTranslatef(x, y, 0);
    glScalef(scale, scale, 1.0f);
    glColor3f(1.0f, 1.0f, 1.0f);

    for (char c : text) {
        glutStrokeCharacter(GLUT_STROKE_ROMAN, c);
    }

    glPopMatrix();
}


int main() {

    int argc = 1;
    char* argv[] = {(char*)"program"};
    glutInit(&argc, argv);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Multi-DOF Spring-Mass System", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    // Set up mouse callbacks (after window is created)
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    // Set up OpenGL
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2.0, 2.0, -1.5, 1.5, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::cout << "Controls:" << std::endl;
    std::cout << "- Press Space to start/stop oscillation" << std::endl;
    std::cout << "- Press R to reset" << std::endl;

    float timeStep = 0.016f; // ~60 FPS

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw grid first (background)
        drawGrid(0.1f, 20);

        // Update physics
        if (oscillationStarted) {
            // If dragging, pause simulation for that block
            std::vector<bool> isDragged(numMasses, false);
            if (draggedBlock != -1) isDragged[draggedBlock] = true;
            std::vector<float> y(2 * numMasses);
            for (int i = 0; i < numMasses; ++i) {
                y[2 * i] = positions[i];
                y[2 * i + 1] = velocities[i];
            }
            // RK4 step for internal springs
            y = multiSystem.step(simTime, y, timeStep);
            for (int i = 0; i < numMasses; ++i) {
                if (!isDragged[i]) {
                    positions[i] = y[2 * i];
                    velocities[i] = y[2 * i + 1];
                }
            }
            // Find the leftmost box (smallest x)
            int leftIdx = 0;
            float minX = positions[0];
            for (int i = 1; i < numMasses; ++i) {
                if (positions[i] < minX) {
                    minX = positions[i];
                    leftIdx = i;
                }
            }
            float xLeft = positions[leftIdx];
            float vLeft = velocities[leftIdx];
            float forceWall = -wallSpringConstant * (xLeft - wallPosition) - dampings[leftIdx] * vLeft;
            float aWall = forceWall / masses[leftIdx];
            velocities[leftIdx] += aWall * timeStep;
            positions[leftIdx] += velocities[leftIdx] * timeStep;

            // Collision handling: prevent overlap and swap velocities if blocks overlap
            float minDist = 0.11f; // slightly more than twice the mass size (0.05)
            for (int i = 0; i < numMasses - 1; ++i) {
                if (positions[i] - positions[i+1] < minDist) {
                    // Move them apart
                    float overlap = minDist - (positions[i] - positions[i+1]);
                    positions[i] += 0.5f * overlap;
                    positions[i+1] -= 0.5f * overlap;
                    // Elastic collision: swap velocities
                    std::swap(velocities[i], velocities[i+1]);
                }
            }
            simTime += timeStep;
        }

        // Draw wall as a thick blue vertical line
        glColor3f(0.2f, 0.2f, 1.0f);
        glLineWidth(8.0f);
        glBegin(GL_LINES);
        glVertex2f(wallPosition, -0.3f);
        glVertex2f(wallPosition, 0.3f);
        glEnd();
        // Draw spring from wall to leftmost box
        int leftIdx = 0;
        float minX = positions[0];
        for (int i = 1; i < numMasses; ++i) {
            if (positions[i] < minX) {
                minX = positions[i];
                leftIdx = i;
            }
        }
        drawSpring(wallPosition, positions[leftIdx], 0.0f);
        // Draw springs between boxes
        for (size_t i = 0; i < couplings.size(); ++i) {
            int idxA = couplings[i].first;
            int idxB = couplings[i].second;
            drawSpring(positions[idxA], positions[idxB], 0.0f);
        }
        // Draw masses
        for (int i = 0; i < numMasses; ++i) {
            drawMass(positions[i], 0.0f, false);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Handle keyboard input
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            oscillationStarted = true;
        }
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            positions = initialPositions;
            velocities = initialVelocities;
            simTime = 0.0f;
            oscillationStarted = false;
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}