
#include "MechanicalSystem.h"
#include <cmath>
#include <matplotlibcpp.h>
#include <Eigen/Dense>

namespace plt = matplotlibcpp;

// Use the correct namespace for Eigen
using Eigen::VectorXf;


MechanicalSystem::MechanicalSystem(float m, float c, float k, const Vector& x0, const Vector& v0)
    : mass(m), damping(c), springConstant(k), initialPosition(x0), initialVelocity(v0), externalForce(Vector::Zero(x0.size())), gravity(Vector::Zero(x0.size())) {}
void MechanicalSystem::setGravity(const Vector& g) {
    gravity = g;
}

float MechanicalSystem::getMass() const { return mass; }
float MechanicalSystem::getDamping() const { return damping; }
float MechanicalSystem::getSpringConstant() const { return springConstant; }
MechanicalSystem::Vector MechanicalSystem::getInitialPosition() const { return initialPosition; }
MechanicalSystem::Vector MechanicalSystem::getInitialVelocity() const { return initialVelocity; }

void MechanicalSystem::setMass(float m) { mass = m; }
void MechanicalSystem::setStiffness(float k) { springConstant = k; }
void MechanicalSystem::setDamping(float c) { damping = c; }

MechanicalSystem::Vector MechanicalSystem::positionDerivative(float /*t*/, const Vector& /*x*/, const Vector& v) const {
    return v;
}

MechanicalSystem::Vector MechanicalSystem::velocityDerivative(float /*t*/, const Vector& x, const Vector& v) const {
    // F = -kx - cv + externalForce + m*gravity
    return (-getSpringConstant() * x - getDamping() * v + externalForce + mass * gravity) / getMass();
}

MechanicalSystem::Vector MechanicalSystem::getVelocity(float t, const Vector& x, const Vector& v, float h) {
    Vector y0(x.size() + v.size());
    y0 << x, v;
    Vector result = rk4(systemOde, t, y0, h, *this);
    return result.tail(v.size());
}

MechanicalSystem::Vector MechanicalSystem::getPos(float t, const Vector& x, const Vector& v, float h) {
    Vector y0(x.size() + v.size());
    y0 << x, v;
    Vector result = rk4(systemOde, t, y0, h, *this);
    return result.head(x.size());
}

void MechanicalSystem::updateExternalForce(const Vector& force) {
    externalForce = force;
}

MechanicalSystem::Vector MechanicalSystem::systemOde(float t, const Vector& y, const MechanicalSystem& system) {
    int n = y.size() / 2;
    Vector x = y.head(n);
    Vector v = y.tail(n);
    Vector dx = system.positionDerivative(t, x, v);
    Vector dv = system.velocityDerivative(t, x, v);
    Vector dydt(y.size());
    dydt << dx, dv;
    return dydt;
}

MechanicalSystem::Vector MechanicalSystem::rk4(Vector (*f)(float, const Vector&, const MechanicalSystem&), float t, const Vector& y, float h, const MechanicalSystem& system) {
    Vector k1 = f(t, y, system);
    Vector k2 = f(t + h / 2, y + h / 2 * k1, system);
    Vector k3 = f(t + h / 2, y + h / 2 * k2, system);
    Vector k4 = f(t + h, y + h * k3, system);
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

void MechanicalSystem::createPlot(float T) {
    float t = 0.0;
    float h = 0.1;
    Vector y = initialPosition;
    Vector v = initialVelocity;

    std::vector<float> timePoints;
    std::vector<float> positionPoints;
    std::vector<float> velocityPoints;

    while (t <= T) {
        Vector currentPos = getPos(t, y, v, h);
        Vector currentVel = getVelocity(t, y, v, h);

        t += h;
        y = currentPos;
        v = currentVel;
        // For plotting, only plot the first dimension
        positionPoints.push_back(currentPos[0]);
        velocityPoints.push_back(currentVel[0]);
        timePoints.push_back(t);
    }

    plt::plot(timePoints, positionPoints, {{"label", "Position (dim 0)"}});
    plt::plot(timePoints, velocityPoints, {{"label", "Velocity (dim 0)"}});
    plt::xlabel("Time");
    plt::ylabel("Value");
    plt::title("Trajectory of the Mass (First Dimension)");
    plt::legend();
    plt::save("spring_figure.png");
}
