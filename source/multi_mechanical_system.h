#ifndef NDOF_SYSTEM_H
#define NDOF_SYSTEM_H

#include <vector>
#include "mechanical_system.h" // Include the header for MechanicalSystem

class NDOFSystem {
private:
    std::vector<MechanicalSystem> systems;
    std::vector<std::vector<float>> couplingForces;
    int numDOF;

public:
    NDOFSystem(const std::vector<MechanicalSystem>& systemInstances,
               const std::vector<std::vector<float>>& couplingForcesMatrix);

    void updateSystems(float T);
    void createCoupling(const std::vector<std::vector<float>>& couplingMatrix);

    Eigen::MatrixXf getMassMatrix() const;
    Eigen::MatrixXf getDampingMatrix() const;
    Eigen::MatrixXf getStiffnessMatrix() const;
    Eigen::VectorXf getNaturalFrequency() const;
};

#endif // NDOF_SYSTEM_H
