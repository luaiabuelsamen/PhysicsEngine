#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>

#include "MultiMechanicalSystem.h"
#include "cuda_mass_spring.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declarations
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
float screenToWorldX(double screenX, int windowWidth);
bool isMouseOverCube(float mouseWorldX, float cubeWorldX);
bool isMouseOverCube(float mouseWorldX, const Eigen::VectorXf& cubeWorldX);
void drawMass(float xPos, float yPos = 0.0f, bool isHovered = false);
void drawSpring(float x1, float x2, float y = 0.0f);
void drawGrid(float spacing, int numLines);
void drawText(const std::string& text, float x, float y, float scale = 1.0f);

#ifdef USE_CUDA
// Helper to call CUDA coupled mass-spring kernel for one step
void runCudaCoupledMassSpring(
    std::vector<float>& positions,
    std::vector<float>& velocities,
    const std::vector<float>& initialPositions,
    const std::vector<float>& initialVelocities,
    const std::vector<float>& masses,
    const std::vector<float>& dampings,
    const std::vector<float>& springConstants,
    const std::vector<std::pair<int, int>>& couplings,
    const std::vector<float>& couplingConstants,
    int steps, float h)
{
    int numMasses = positions.size();
    int numCouplings = couplings.size();
    std::vector<int> couplingA(numCouplings), couplingB(numCouplings);
    for (int i = 0; i < numCouplings; ++i) {
        couplingA[i] = couplings[i].first;
        couplingB[i] = couplings[i].second;
    }
    cuda_coupled_mass_spring(
        positions.data(), velocities.data(),
        initialPositions.data(), initialVelocities.data(),
        masses.data(), dampings.data(), springConstants.data(),
        couplingA.data(), couplingB.data(), couplingConstants.data(), numCouplings,
        numMasses, steps, h);
}
#else
// Stub for non-CUDA builds
void runCudaCoupledMassSpring(
    std::vector<float>&, std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
    const std::vector<std::pair<int, int>>&, const std::vector<float>&,
    int, float)
{
    // No-op in non-CUDA build
}
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

// Wall spring constant
float wallSpringConstant = 1.0f;
float wallPosition = -1.0f;

// Global variables
MultiMechanicalSystem multiSystem(masses, dampings, springConstants, initialPositions, initialVelocities, couplings, couplingConstants);
std::vector<float> positions = initialPositions;
std::vector<float> velocities = initialVelocities;
float simTime = 0.0f;
bool oscillationStarted = true;
int draggedBlock = -1; // index of block being dragged, -1 if none
float dragOffset = 0.0f;
double lastMouseX = 0.0;

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

// Mouse event handlers
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
    std::cout << "- Click and drag masses to interact" << std::endl;

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


#ifdef USE_CUDA
            // CUDA step for internal springs
            std::vector<float> newPositions = positions;
            std::vector<float> newVelocities = velocities;
            runCudaCoupledMassSpring(
                newPositions, newVelocities,
                positions, velocities,
                masses, dampings, springConstants,
                couplings, couplingConstants,
                1, timeStep);
            // Update positions and velocities for non-dragged masses
            for (int i = 0; i < numMasses; ++i) {
                if (!isDragged[i]) {
                    positions[i] = newPositions[i];
                    velocities[i] = newVelocities[i];
                }
            }
#else
            // CPU RK4 step for internal springs
            std::vector<float> y(2 * numMasses);
            for (int i = 0; i < numMasses; ++i) {
                y[2 * i] = positions[i];
                y[2 * i + 1] = velocities[i];
            }
            y = multiSystem.step(simTime, y, timeStep);
            for (int i = 0; i < numMasses; ++i) {
                if (!isDragged[i]) {
                    positions[i] = y[2 * i];
                    velocities[i] = y[2 * i + 1];
                }
            }
#endif

            // Find the leftmost mass (smallest x coordinate)
            int leftIdx = 0;
            for (int i = 1; i < numMasses; ++i) {
                if (positions[i] < positions[leftIdx]) {
                    leftIdx = i;
                }
            }

            // Apply wall force to leftmost mass
            if (!isDragged[leftIdx]) {
                float xLeft = positions[leftIdx];
                float vLeft = velocities[leftIdx];
                float forceWall = -wallSpringConstant * (xLeft - wallPosition) - dampings[leftIdx] * vLeft;
                float aWall = forceWall / masses[leftIdx];
                velocities[leftIdx] += aWall * timeStep;
                positions[leftIdx] += velocities[leftIdx] * timeStep;
            }

            // Collision handling: prevent overlap between adjacent masses
            float minDist = 0.11f; // slightly more than twice the mass size (0.05)
            for (int i = 0; i < numMasses - 1; ++i) {
                for (int j = i + 1; j < numMasses; ++j) {
                    float dist = std::abs(positions[i] - positions[j]);
                    if (dist < minDist) {
                        // Calculate overlap and separate masses
                        float overlap = minDist - dist;
                        float direction = (positions[i] > positions[j]) ? 1.0f : -1.0f;
                        
                        if (!isDragged[i]) positions[i] += 0.5f * overlap * direction;
                        if (!isDragged[j]) positions[j] -= 0.5f * overlap * direction;
                        
                        // Elastic collision: exchange velocities
                        if (!isDragged[i] && !isDragged[j]) {
                            std::swap(velocities[i], velocities[j]);
                        }
                    }
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

        // Draw spring from wall to leftmost mass
        int leftIdx = 0;
        for (int i = 1; i < numMasses; ++i) {
            if (positions[i] < positions[leftIdx]) {
                leftIdx = i;
            }
        }
        drawSpring(wallPosition, positions[leftIdx], 0.0f);

        // Draw springs between coupled masses
        for (size_t i = 0; i < couplings.size(); ++i) {
            int idxA = couplings[i].first;
            int idxB = couplings[i].second;
            drawSpring(positions[idxA], positions[idxB], 0.0f);
        }

        // Draw masses
        for (int i = 0; i < numMasses; ++i) {
            bool isHovered = (draggedBlock == i);
            drawMass(positions[i], 0.0f, isHovered);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        // Handle keyboard input
        static bool spacePressed = false;
        static bool rPressed = false;
        
        bool currentSpaceState = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);
        bool currentRState = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
        
        if (currentSpaceState && !spacePressed) {
            oscillationStarted = !oscillationStarted;
        }
        
        if (currentRState && !rPressed) {
            positions = initialPositions;
            velocities = initialVelocities;
            simTime = 0.0f;
            oscillationStarted = false;
            draggedBlock = -1;
        }
        
        spacePressed = currentSpaceState;
        rPressed = currentRState;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}