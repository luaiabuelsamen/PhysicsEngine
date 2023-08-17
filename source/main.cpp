#include "mechanical_system.h"
#include "multi_mechanical_system.h"
#include <Eigen/Dense>
#include <iostream>

int main() {
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
