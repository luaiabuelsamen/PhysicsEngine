#include "mechanical_system.h"
#include "multi_mechanical_system.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // mass(m), damping(c), springConstant(k), externalForce(), initialPosition(x0), initialVelocity(v0)
    // Create instances of MechanicalSystem for each cart (mass-spring-damper)
    MechanicalSystem cart1(1.0, 0.1, 1.0, 0.0, 0.0, 0.0);
    MechanicalSystem cart2(1.0, 0.1, 1.0, 0.0, 0.0, 0.0);
    MechanicalSystem cart3(1.0, 0.1, 1.0, 0.0, 0.0, 0.0);

    // // Combine instances into an NDOFSystem
    // std::vector<MechanicalSystem> systemInstances = {cart1, cart2, cart3};
    
    // Create a coupling matrix for the 3-DOF system
    std::vector<std::vector<float>> couplingMatrix = {
        {0.0, 1.0, 0.0}, // Coupling between cart 1 and cart 2
        {1.0, 0.0, 1.0}, // Coupling between cart 2 and cart 1 and cart 3
        {0.0, 1.0, 0.0}  // Coupling between cart 3 and cart 2
    };
    // Parameters for rapid oscillation
    float mass = 1.0;           // Mass of the system
    float damping = 0.1;        // Damping coefficient (lower values lead to more oscillation)
    float springConstant = 2.0; // Spring constant (higher values lead to more oscillation)
    float externalForce = 0.0;  // External force (set to 0 for free oscillation)
    float initialPosition = 2.0;// Initial position
    float initialVelocity = 0.0; // Initial velocity

    MechanicalSystem rapidOscillatingSystem(mass, damping, springConstant, externalForce, initialPosition, initialVelocity);



    rapidOscillatingSystem.createPlot(30);
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
