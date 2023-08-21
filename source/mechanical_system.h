#ifndef MECHANICAL_SYSTEM_H
#define MECHANICAL_SYSTEM_H

#include <functional>
#include <vector>

class MechanicalSystem {
private:
    float mass;
    float damping;
    float springConstant;
    float initialPosition;
    float initialVelocity;
public:
    float externalForce;
    MechanicalSystem(float m, float c, float k, float F_ext, float x0, float v0);
    float getMass() const;
    float getDamping() const;
    float getSpringConstant() const;
    float getInitialPosition() const;
    float getInitialVelocity() const;

    float getPos(float t, float x0, float y0, float h) const;
    float getVelocity(float t, float x0, float y0, float h) const;
    float positionDerivative(float t, float x, float v) const;
    float velocityDerivative(float t, float x, float v) const;
    static std::vector<double> systemOde(float t, std::vector<double> y, const MechanicalSystem& system);
    static std::vector<double> rk4(std::vector<double> (*f)(float, std::vector<double>, const MechanicalSystem&), float t, const std::vector<double>& y, float h, const MechanicalSystem& system);

    void updateExternalForce(float forceFunction, float T);
    void createPlot(float T);
};

#endif // MECHANICAL_SYSTEM_H
