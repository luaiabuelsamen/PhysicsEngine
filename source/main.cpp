#include "MechanicalSystem.h"
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <sstream>

struct SystemState {
    float position = 0.0f;
    float velocity = 0.0f;
    float time = 0.0f;
    bool isDragging = false;
    float dragStartPos = 0.0f;
    float lastMouseX = 0.0f;
    bool oscillationStarted = false;
    std::chrono::steady_clock::time_point lastClickTime = std::chrono::steady_clock::now();
    bool isEditingBlock = false;
    bool isEditingSpring = false;
    float hoverAlpha = 0.0f;
    
    std::string inputBuffer;
    bool isEditingMass = false;
    bool isEditingDamping = false;
    bool isEditingStiffness = false;
    float currentMass = 1.0f;
    float currentDamping = 0.1f;
    float currentStiffness = 1.0f;
    
    MechanicalSystem system1d;
    
    SystemState()
        : system1d(1.0f, 0.1f, 1.0f, position, velocity) {}
};

// Forward declarations
float screenToWorldX(double screenX, int windowWidth);
bool isMouseOverCube(float mouseWorldX, float cubeWorldX);
void drawMass(float xPos, bool isHovered);

SystemState state;

// Simple text rendering using GL_LINES
void renderText(const std::string& text, float x, float y, float scale = 0.001f) {
    glPushMatrix();
    glTranslatef(x, y, 0.0f);
    glScalef(scale, scale, 1.0f);
    glLineWidth(2.0f);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    
    for (char c : text) {
        // Simple line-based characters
        switch(c) {
            case '0': // Draw a box
                glBegin(GL_LINE_LOOP);
                glVertex2f(0, 0);
                glVertex2f(50, 0);
                glVertex2f(50, 100);
                glVertex2f(0, 100);
                glEnd();
                glTranslatef(60, 0, 0);
                break;
            case '.':
                glBegin(GL_POINTS);
                glVertex2f(0, 0);
                glEnd();
                glTranslatef(20, 0, 0);
                break;
            default: // For other characters, just draw a vertical line
                glBegin(GL_LINES);
                glVertex2f(0, 0);
                glVertex2f(0, 100);
                glEnd();
                glTranslatef(40, 0, 0);
        }
    }
    glPopMatrix();
    glLineWidth(1.0f);
}

float screenToWorldX(double screenX, int windowWidth) {
    return (float)(screenX - windowWidth / 2) / (windowWidth / 4);
}

bool isMouseOverCube(float mouseWorldX, float cubeWorldX) {
    float cubeSize = 0.05f;
    return std::abs(mouseWorldX - cubeWorldX) < cubeSize;
}

void drawInputField(float x, float y, const std::string& label, const std::string& value, bool isActive) {
    // Draw a smaller, semi-transparent background
    glBegin(GL_QUADS);
    if (isActive) {
        glColor4f(0.3f, 0.3f, 0.5f, 0.9f);  // More visible when active
    } else {
        glColor4f(0.2f, 0.2f, 0.3f, 0.8f);
    }
    float width = 0.6f;
    float height = 0.08f;
    glVertex2f(x - width/2, y - height/2);
    glVertex2f(x + width/2, y - height/2);
    glVertex2f(x + width/2, y + height/2);
    glVertex2f(x - width/2, y + height/2);
    glEnd();

    // Draw border for active field
    if (isActive) {
        glBegin(GL_LINE_LOOP);
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        glVertex2f(x - width/2, y - height/2);
        glVertex2f(x + width/2, y - height/2);
        glVertex2f(x + width/2, y + height/2);
        glVertex2f(x - width/2, y + height/2);
        glEnd();
    }

    // Draw simple text using lines
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    std::string displayText = label + ": " + value;
    
    // Using OpenGL lines to draw text
    glBegin(GL_LINES);
    float charWidth = 0.02f;
    float charHeight = 0.05f;
    float xPos = x - width/2 + 0.05f;
    for (char c : displayText) {
        // Simple line representation of characters
        if (c != ' ') {
            glVertex2f(xPos, y - charHeight/2);
            glVertex2f(xPos, y + charHeight/2);
        }
        xPos += charWidth;
    }
    glEnd();
}

void drawPropertyEditor() {
    if (state.isEditingBlock || state.isEditingSpring) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Overlay background
        glBegin(GL_QUADS);
        glColor4f(0.0f, 0.0f, 0.0f, 0.5f);
        glVertex2f(-2.0f, -1.5f);
        glVertex2f(2.0f, -1.5f);
        glVertex2f(2.0f, 1.5f);
        glVertex2f(-2.0f, 1.5f);
        glEnd();

        // Draw input fields
        drawInputField(0.0f, 0.6f, "Mass",
                       state.isEditingMass ? state.inputBuffer : std::to_string(state.currentMass),
                       state.isEditingMass);
        drawInputField(0.0f, 0.4f, "Damping",
                       state.isEditingDamping ? state.inputBuffer : std::to_string(state.currentDamping),
                       state.isEditingDamping);
        drawInputField(0.0f, 0.2f, "Stiffness",
                       state.isEditingStiffness ? state.inputBuffer : std::to_string(state.currentStiffness),
                       state.isEditingStiffness);

        glDisable(GL_BLEND);
    }
}

// Update the mouseButtonCallback to close the editor when clicking outside
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    double mouseX, mouseY;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    float mouseWorldX = screenToWorldX(mouseX, width);

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - state.lastClickTime).count();
        state.lastClickTime = now;

        if (isMouseOverCube(mouseWorldX, state.position)) {
            if (duration < 300) {  // Double click
                state.isEditingBlock = true;
                state.isEditingMass = true;
                state.inputBuffer = std::to_string(state.currentMass);
            } else {
                state.isDragging = true;
                state.dragStartPos = mouseWorldX;
                state.lastMouseX = mouseWorldX;
                state.oscillationStarted = false;
                state.velocity = 0.0f;
            }
        } else {
            // Click outside the cube - close the editor if it's open
            if (state.isEditingBlock || state.isEditingSpring) {
                state.isEditingBlock = false;
                state.isEditingSpring = false;
                state.isEditingMass = false;
                state.isEditingDamping = false;
                state.isEditingStiffness = false;
                state.inputBuffer.clear();
            }
        }
    } else if (action == GLFW_RELEASE && state.isDragging) {
        state.isDragging = false;
        state.oscillationStarted = true;
        state.velocity = (mouseWorldX - state.lastMouseX) / 0.016f;
        state.time = 0.0f;
    }
}

void drawMass(float xPos, bool isHovered) {
    glBegin(GL_QUADS);
    float size = 0.05f;
    if (isHovered) {
        glColor4f(1.0f, 0.5f, 0.3f, 1.0f);
    } else {
        glColor4f(0.9f, 0.3f, 0.2f, 1.0f);
    }
    glVertex2f(xPos - size, -size);
    glVertex2f(xPos + size, -size);
    glVertex2f(xPos + size, size);
    glVertex2f(xPos - size, size);
    glEnd();
}

void drawSpring(float xPos) {
    glBegin(GL_LINE_STRIP);
    glColor4f(0.4f, 0.8f, 1.0f, 1.0f);
    
    int segments = 30;
    float amplitude = 0.15f;
    float stretch = std::abs(xPos);
    
    for (int i = 0; i <= segments; i++) {
        float t = (float)i / segments;
        float x = (xPos * t);
        float y = amplitude * sin(t * 10 * M_PI) * (1.0f - 0.5f * std::abs(t - 0.5f));
        glVertex2f(x, y);
    }
    glEnd();
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (state.isEditingMass || state.isEditingDamping || state.isEditingStiffness) {
            if (key == GLFW_KEY_ENTER) {
                try {
                    float value = std::stof(state.inputBuffer);
                    if (state.isEditingMass) {
                        state.currentMass = value;
                        state.system1d.setMass(value);
                    } else if (state.isEditingDamping) {
                        state.currentDamping = value;
                        state.system1d.setDamping(value);
                    } else if (state.isEditingStiffness) {
                        state.currentStiffness = value;
                        state.system1d.setStiffness(value);
                    }
                } catch (...) {
                    // Invalid input, ignore
                }
                state.isEditingMass = state.isEditingDamping = state.isEditingStiffness = false;
                state.inputBuffer.clear();
            } else if (key == GLFW_KEY_BACKSPACE && !state.inputBuffer.empty()) {
                state.inputBuffer.pop_back();
            } else if (key >= GLFW_KEY_0 && key <= GLFW_KEY_9 || key == GLFW_KEY_PERIOD) {
                char c = (key == GLFW_KEY_PERIOD) ? '.' : ('0' + (key - GLFW_KEY_0));
                state.inputBuffer += c;
            }
        }
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (state.isDragging) {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        float mouseWorldX = screenToWorldX(xpos, width);
        state.position = mouseWorldX;
        state.lastMouseX = mouseWorldX;
    }
}

void drawGrid(float spacing, int numLines) {
    glColor3f(0.8f, 0.8f, 0.8f); // Light gray color for the grid
    glBegin(GL_LINES);

    // Vertical lines
    for (int i = -numLines; i <= numLines; ++i) {
        float x = i * spacing;
        glVertex2f(x, -1.0f); // Start of the line
        glVertex2f(x, 1.0f);  // End of the line
    }

    // Horizontal lines
    for (int i = -numLines; i <= numLines; ++i) {
        float y = i * spacing;
        glVertex2f(-1.0f, y); // Start of the line
        glVertex2f(1.0f, y);  // End of the line
    }

    glEnd();
}

void drawText(const std::string& text, float x, float y, float scale = 1.0f) {
    glPushMatrix();
    glTranslatef(x, y, 0);
    glScalef(scale, scale, 1.0f);

    for (char c : text) {
        glutStrokeCharacter(GLUT_STROKE_ROMAN, c);
    }

    glPopMatrix();
}

void drawMassProperties() {
    std::string massText = "Mass: " + std::to_string(state.currentMass);
    std::string dampingText = "Damping: " + std::to_string(state.currentDamping);
    std::string stiffnessText = "Stiffness: " + std::to_string(state.currentStiffness);
    std::string positionText = "Position: " + std::to_string(state.position);

    float x = 0.7f; // Top-right corner X position
    float y = 0.9f; // Start Y position
    
    // Draw each property
    drawText(massText, x, y, 0.001f);
    drawText(dampingText, x, y - 0.2f, 0.001f);
    drawText(stiffnessText, x, y - 0.4f, 0.001f);
    drawText(positionText, x, y - 0.6f, 0.001f);
}


int main() {
    int argc = 1;
    char* argv[1];
    glutInit(&argc, argv);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "Spring-Mass System", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glOrtho(-2.0f, 2.0f, -1.5f, 1.5f, -1.0f, 1.0f);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetKeyCallback(window, keyCallback);
    drawGrid(0.1f, 20);


    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (!state.isDragging && state.oscillationStarted) {
            float timeStep = 0.016f;
            state.position = state.system1d.getPos(state.time, state.position, state.velocity, timeStep);
            state.velocity = state.system1d.getVelocity(state.time, state.position, state.velocity, timeStep);
            state.time += timeStep;
        }

        drawMassProperties();
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        float mouseWorldX = screenToWorldX(mouseX, width);

        drawSpring(state.position);
        drawMass(state.position, isMouseOverCube(mouseWorldX, state.position));
        drawPropertyEditor();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}