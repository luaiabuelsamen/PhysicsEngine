#include <vector>
#include "MechanicalSystem.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

class NDOFSystem {
private:
    std::vector<MechanicalSystem> systems;
    std::vector<std::vector<float>> couplingForces;
    int numDOF;

public:
    NDOFSystem(const std::vector<MechanicalSystem>& systemInstances,
               const std::vector<std::vector<float>>& couplingForcesMatrix)
        : systems(systemInstances), couplingForces(couplingForcesMatrix), numDOF(systemInstances.size()) {
        // Ensure that the coupling forces matrix is correctly sized
        if (couplingForcesMatrix.size() != numDOF || couplingForcesMatrix[0].size() != numDOF) {
            throw std::runtime_error("Invalid size of coupling forces matrix.");
        }
    }

    Eigen::MatrixXf getMassMatrix() const {
        Eigen::MatrixXf massMatrix = Eigen::MatrixXf::Zero(numDOF, numDOF);

        for (int i = 0; i < numDOF; ++i) {
            massMatrix(i, i) = systems[i].getMass();
        }

        return massMatrix;
    }

    // Method to calculate the damping matrix
    Eigen::MatrixXf getDampingMatrix() const {
        Eigen::MatrixXf dampingMatrix = Eigen::MatrixXf::Zero(numDOF, numDOF);

        for (int i = 0; i < numDOF; ++i) {
            dampingMatrix(i, i) = systems[i].getDamping();
        }

        return dampingMatrix;
    }

    // Method to calculate the stiffness matrix
    Eigen::MatrixXf getStiffnessMatrix() const {
        Eigen::MatrixXf stiffnessMatrix = Eigen::MatrixXf::Zero(numDOF, numDOF);

        for (int i = 0; i < numDOF; ++i) {
            stiffnessMatrix(i, i) = systems[i].getSpringConstant();
        }

        return stiffnessMatrix;
    }

    // Method to calculate the natural frequency of the system using SVD
    Eigen::VectorXf getNaturalFrequency() const {
        Eigen::MatrixXf massMatrix = getMassMatrix();
        Eigen::MatrixXf stiffnessMatrix = getStiffnessMatrix();

        Eigen::MatrixXf A = massMatrix.inverse() * stiffnessMatrix;
        Eigen::EigenSolver<Eigen::MatrixXf> eigenSolver(A);
        Eigen::VectorXf eigenvalues = eigenSolver.eigenvalues().real();
        Eigen::VectorXf naturalFrequency = eigenvalues.array().sqrt();

        return naturalFrequency;
    }
};
