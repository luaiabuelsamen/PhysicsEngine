#ifndef MECHANICAL_SYSTEM_H
#define MECHANICAL_SYSTEM_H

#include <functional>
#include <vector>

class MechanicalSystem {
private:
    float mass;
    float damping;
    float springConstant;
    float externalForce;
    float initialPosition;
    float initialVelocity;

public:
    MechanicalSystem(float m, float c, float k, float F_ext, float x0, float v0);

    float getMass() const;
    float getDamping() const;
    float getSpringConstant() const;
    float getInitialPosition() const;
    float getInitialVelocity() const;

    float getPos(float T) const;
    float getVelocity(float T) const;

    void updateExternalForce(std::function<float(float)> forceFunction, float T);
    void createPlot(float T) const;
};

#endif // MECHANICAL_SYSTEM_H
