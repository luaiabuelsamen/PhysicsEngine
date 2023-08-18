#include "mechanical_system.h"
#include "multi_mechanical_system.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // mass(m), damping(c), springConstant(k), externalForce(), initialPosition(x0), initialVelocity(v0)
    // Create instances of MechanicalSystem for each cart (mass-spring-damper)
    MechanicalSystem cart1(1.0, 0.1, 1.0, 0.0, 0.0, 0.0); // Mass, Damping, Spring, External Force, Initial Position, Initial Velocity
    MechanicalSystem cart2(1.0, 0.1, 1.0, 0.0, 0.0, 0.0);
    MechanicalSystem cart3(1.0, 0.1, 1.0, 0.0, 0.0, 0.0);

    // Combine instances into an NDOFSystem
    std::vector<MechanicalSystem> systemInstances = {cart1, cart2, cart3};
    
    // Create a coupling matrix for the 3-DOF system
    std::vector<std::vector<float>> couplingMatrix = {
        {0.0, 1.0, 0.0}, // Coupling between cart 1 and cart 2
        {1.0, 0.0, 1.0}, // Coupling between cart 2 and cart 1 and cart 3
        {0.0, 1.0, 0.0}  // Coupling between cart 3 and cart 2
    };
    MechanicalSystem overdampedSystem(1.0, 2.0, 1.0, 0.0, 2.0, 0.0);

    auto rk4 = [](float (*f)(float, float), float t, float y, float h) -> float {
        float k1 = h * f(t, y);
        float k2 = h * f(t + 0.5 * h, y + 0.5 * k1);
        float k3 = h * f(t + 0.5 * h, y + 0.5 * k2);
        float k4 = h * f(t + h, y + k3);
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
    };

    // Define the ODE function
    auto f = [](float t, float y) -> float {
        // Define your ODE here: y'' = -2*y' - y
        // For this example, let's use the given equation: y'' = -2*y' - y
        return -2.0 * y - 0.0;  // You should include the proper calculations.
    };

    // Initial conditions
    float t = 0.0;  // Initial time
    float y = 2.0;  // Initial position
    float h = 0.1;  // Step size

    // Time loop
    while (t <= 10.0) {  // Adjust the end time as needed
        y = rk4(f, t, y, h);
        t += h;
        std::cout << "Time: " << t << " Position: " << y << std::endl;
    }


    // overdampedSystem.createPlot(10);
    // NDOFSystem ndofSystem(systemInstances, couplingMatrix);

    // Apply an impulse force to the first cart at t = 0.5 seconds
    // ndofSystem.updateSystems(0.5);

    // // // Get the combined positions and velocities of the system at t = 0.5 seconds
    // // std::vector<float> combinedPositions = ndofSystem.getCombinedPositions(0.5);
    // // std::vector<float> combinedVelocities = ndofSystem.getCombinedVelocities(0.5);

    // // Calculate and print the mass, damping, and stiffness matrices
    // Eigen::MatrixXf massMatrix = ndofSystem.getMassMatrix();
    // Eigen::MatrixXf dampingMatrix = ndofSystem.getDampingMatrix();
    // Eigen::MatrixXf stiffnessMatrix = ndofSystem.getStiffnessMatrix();

    // std::cout << "Mass Matrix:\n" << massMatrix << std::endl;
    // std::cout << "Damping Matrix:\n" << dampingMatrix << std::endl;
    // std::cout << "Stiffness Matrix:\n" << stiffnessMatrix << std::endl;

    // // Calculate and print the natural frequencies of the system
    // Eigen::VectorXf naturalFrequency = ndofSystem.getNaturalFrequency();
    // std::cout << "Natural Frequencies:\n" << naturalFrequency << std::endl;

    return 0;
}
