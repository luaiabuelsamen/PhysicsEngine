#include "ode.h"
#include <iostream>
#include <cmath>
#include <functional>
#include "mechanical_system.h"
// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;

using namespace std;
using namespace placeholders;
auto positionDerivative = [](const MechanicalSystem& system, float t, float pos) {
    return system.getInitialVelocity();  // Return velocity as the derivative of position
};

// Define the derivative function for velocity using a lambda
auto velocityDerivative = [](const MechanicalSystem& system, float t, float vel) {
    // Calculate acceleration based on damping, spring force, mass, and external force
    float acceleration = -(system.getDamping() / system.getMass()) * vel -
                         (system.getSpringConstant() / system.getMass()) * system.getInitialPosition() +
                         (system.externalForce / system.getMass());
    return acceleration;
};

MechanicalSystem::MechanicalSystem(float m, float c, float k, float F_ext, float x0, float v0)
    : mass(m), damping(c), springConstant(k), externalForce(), initialPosition(x0), initialVelocity(v0) {}

// Getter methods
float MechanicalSystem::getMass() const { return mass; }
float MechanicalSystem::getDamping() const { return damping; }
float MechanicalSystem::getSpringConstant() const { return springConstant; }
float MechanicalSystem::getInitialPosition() const { return initialPosition; }
float MechanicalSystem::getInitialVelocity() const { return initialVelocity; }


float MechanicalSystem::positionDerivative(float t, float pos, float initialVelocity) {
    return initialVelocity;  // Return velocity as the derivative of position
}

float MechanicalSystem::velocityDerivative(float t, float vel, float damping, float mass, float springConstant, float initialPosition, float externalForce) {
    // Calculate acceleration based on damping, spring force, mass, and external force
    float acceleration = -(damping / mass) * vel - (springConstant / mass) * initialPosition + (externalForce / mass);
    return acceleration;
}

float MechanicalSystem::getPos(float T) const {
    float x0 = 0.0;  // Initial time
    float y0 = initialPosition;
    int n = static_cast<int>(abs(T - x0));  // Number of steps

    return rk4(x0, y0, T, n, positionDerivative, initialVelocity);
}

float MechanicalSystem::getVelocity(float T) const {
    float x0 = 0.0;
    float y0 = initialVelocity;
    int n = static_cast<int>(abs(T - x0));

    return rk4(x0, y0, T, n, velocityDerivative, damping, mass, springConstant, initialPosition, externalForce);
}

// Method to update external force using a given force function
void MechanicalSystem::updateExternalForce(float forceFunction, float T) {
    externalForce = forceFunction;
}

// void createPlot(float T) const {
//     // Create a figure and plot the trajectory
//     cppplotlib::figure fig;
//     cppplotlib::plot plt;

//     // Generate points for the trajectory plot
//     std::vector<float> timePoints;
//     std::vector<float> positionPoints;

//     float timeStep = 0.01;  // Time step for generating points
//     float currentPosition = initialPosition; // Initialize position with the initial position
//     for (float t = 0.0; t <= T; t += timeStep) {
//         timePoints.push_back(t);
//         currentPosition = getPos(t); // Calculate position using the updated getPos method
//         positionPoints.push_back(currentPosition);
//     }

//     plt.plot(timePoints, positionPoints);
//     plt.xlabel("Time");
//     plt.ylabel("Position");
//     plt.title("Trajectory of the Mass");
//     fig.show();
// }

