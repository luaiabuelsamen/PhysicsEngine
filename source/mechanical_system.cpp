#include "ode.h"
#include <iostream>
#include <cmath>
#include <functional>
#include "mechanical_system.h"
// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;

using namespace std;

class MechanicalSystem {
private:
    float mass;
    float damping;
    float springConstant;
    float externalForce; // New member variable for external force
    float initialPosition;
    float initialVelocity;

public:
    MechanicalSystem(float m, float c, float k, float F_ext, float x0, float v0)
        : mass(m), damping(c), springConstant(k), externalForce(F_ext), initialPosition(x0), initialVelocity(v0) {}

    // Getter methods for the properties
    float getMass() const { return mass; }
    float getDamping() const { return damping; }
    float getSpringConstant() const { return springConstant; }
    float getInitialPosition() const { return initialPosition; }
    float getInitialVelocity() const { return initialVelocity; }


    // Method to calculate position at time T using RK4 solver
    float getPos(float T) const {
        float x0 = 0.0;  // Initial time
        float y0 = initialPosition;
        int n = static_cast<int>(abs(T - x0));  // Number of steps

        // Define the derivative function for position
        function<float(float, float)> f = [this](float t, float pos) {
            return initialVelocity;  // Return velocity as the derivative of position
        };

        return rk4(x0, y0, T, n, f);
    }

    // Method to calculate velocity at time T using RK4 solver
    float getVelocity(float T) const {
        float x0 = 0.0;
        float y0 = initialVelocity;
        int n = static_cast<int>(abs(T - x0));

        function<float(float, float)> f = [this](float t, float vel) {
            // Calculate acceleration based on damping, spring force, mass, and external force
            float acceleration = -(damping / mass) * vel - (springConstant / mass) * initialPosition + (externalForce / mass);

            return acceleration;
        };

        return rk4(x0, y0, T, n, f);
    }

    void updateExternalForce(function<float(float)> forceFunction, float T) {

        // Update the external force using the provided force function
        externalForce = forceFunction(T);
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

};
