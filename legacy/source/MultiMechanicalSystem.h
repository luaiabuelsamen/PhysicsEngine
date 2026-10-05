    // ...existing code...
#ifndef MULTIMECHANICALSYSTEM_H
#define MULTIMECHANICALSYSTEM_H

#include <vector>
#include <functional>

class MultiMechanicalSystem {
public:
    MultiMechanicalSystem(const std::vector<float>& masses, 
                          const std::vector<float>& dampings, 
                          const std::vector<float>& springConstants, 
                          const std::vector<float>& initialPositions, 
                          const std::vector<float>& initialVelocities, 
                          const std::vector<std::pair<int, int>>& couplings, 
                          const std::vector<float>& couplingConstants);

    void simulate(float T, float h);
    void createPlot();

    // Public wrapper for ODE
    std::vector<float> systemOdePublic(float t, const std::vector<float>& y);

    // Public RK4 step for a single integration step
    std::vector<float> step(float t, const std::vector<float>& y, float h);

private:
    int numSystems;
    std::vector<float> masses, dampings, springConstants;
    std::vector<float> initialPositions, initialVelocities;
    std::vector<std::pair<int, int>> couplings;
    std::vector<float> couplingConstants;

    std::vector<std::vector<float>> rk4(std::function<std::vector<float>(float, const std::vector<float>&)> f, 
                                        float t, 
                                        const std::vector<float>& y, 
                                        float h);

    std::vector<float> systemOde(float t, const std::vector<float>& y);

    std::vector<std::vector<float>> positions, velocities;
};

#endif // MULTIMECHANICALSYSTEM_H
