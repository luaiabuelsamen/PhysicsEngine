#include "MultiMechanicalSystem.h"
#include <vector>
// Public RK4 step for a single integration step
std::vector<float> MultiMechanicalSystem::step(float t, const std::vector<float>& y, float h) {
    auto result = rk4([this](float t, const std::vector<float>& y) { return systemOde(t, y); }, t, y, h);
    return result[0];
}

// Public wrapper for ODE
std::vector<float> MultiMechanicalSystem::systemOdePublic(float t, const std::vector<float>& y) {
    return systemOde(t, y);
}
#include <cmath>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

MultiMechanicalSystem::MultiMechanicalSystem(const std::vector<float>& masses, 
                                             const std::vector<float>& dampings, 
                                             const std::vector<float>& springConstants, 
                                             const std::vector<float>& initialPositions, 
                                             const std::vector<float>& initialVelocities, 
                                             const std::vector<std::pair<int, int>>& couplings, 
                                             const std::vector<float>& couplingConstants)
    : masses(masses), dampings(dampings), springConstants(springConstants), 
      initialPositions(initialPositions), initialVelocities(initialVelocities), 
      couplings(couplings), couplingConstants(couplingConstants) {
    numSystems = masses.size();
}

std::vector<float> MultiMechanicalSystem::systemOde(float t, const std::vector<float>& y) {
    std::vector<float> dydt(2 * numSystems, 0.0f);

    for (size_t i = 0; i < numSystems; ++i) {
        float x = y[2 * i];         // Position
        float v = y[2 * i + 1];     // Velocity
        float springForce = -springConstants[i] * x;
        float dampingForce = -dampings[i] * v;

        // Add coupling forces
        float couplingForce = 0.0f;
        for (size_t j = 0; j < couplings.size(); ++j) {
            if (couplings[j].first == i) {
                couplingForce += -couplingConstants[j] * (x - y[2 * couplings[j].second]);
            } else if (couplings[j].second == i) {
                couplingForce += -couplingConstants[j] * (x - y[2 * couplings[j].first]);
            }
        }

        dydt[2 * i] = v; // dx/dt = velocity
        dydt[2 * i + 1] = (springForce + dampingForce + couplingForce) / masses[i];
    }

    return dydt;
}

std::vector<std::vector<float>> MultiMechanicalSystem::rk4(std::function<std::vector<float>(float, const std::vector<float>&)> f, 
                                                            float t, 
                                                            const std::vector<float>& y, 
                                                            float h) {
    std::vector<float> k1 = f(t, y);
    std::vector<float> y_temp = y;

    for (size_t i = 0; i < y.size(); ++i)
        y_temp[i] = y[i] + 0.5f * h * k1[i];

    std::vector<float> k2 = f(t + 0.5f * h, y_temp);
    for (size_t i = 0; i < y.size(); ++i)
        y_temp[i] = y[i] + 0.5f * h * k2[i];

    std::vector<float> k3 = f(t + 0.5f * h, y_temp);
    for (size_t i = 0; i < y.size(); ++i)
        y_temp[i] = y[i] + h * k3[i];

    std::vector<float> k4 = f(t + h, y_temp);

    std::vector<float> y_next(y.size());
    for (size_t i = 0; i < y.size(); ++i)
        y_next[i] = y[i] + (h / 6.0f) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);

    return {y_next};
}

void MultiMechanicalSystem::simulate(float T, float h) {
    size_t steps = T / h;
    std::vector<float> y(2 * numSystems);
    positions.assign(numSystems, std::vector<float>(steps));
    velocities.assign(numSystems, std::vector<float>(steps));

    for (size_t i = 0; i < numSystems; ++i) {
        y[2 * i] = initialPositions[i];
        y[2 * i + 1] = initialVelocities[i];
        positions[i][0] = initialPositions[i];
        velocities[i][0] = initialVelocities[i];
    }

    for (size_t step = 1; step < steps; ++step) {
        float t = step * h;
        auto nextState = rk4([this](float t, const std::vector<float>& y) { return systemOde(t, y); }, t, y, h);
        y = nextState[0];

        for (size_t i = 0; i < numSystems; ++i) {
            positions[i][step] = y[2 * i];
            velocities[i][step] = y[2 * i + 1];
        }
    }
}

void MultiMechanicalSystem::createPlot() {
    for (size_t i = 0; i < numSystems; ++i) {
        plt::plot(positions[i], {{"label", "System " + std::to_string(i)}});
    }
    plt::legend();
    plt::show();
}
