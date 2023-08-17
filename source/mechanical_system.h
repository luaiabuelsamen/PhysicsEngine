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
    static float positionDerivative(float t, float pos, float initialVelocity);
    static float velocityDerivative(float t, float vel, float damping, float mass, float springConstant, float initialPosition, float externalForce);
public:
    float externalForce;
    MechanicalSystem(float m, float c, float k, float F_ext, float x0, float v0);

    float getMass() const;
    float getDamping() const;
    float getSpringConstant() const;
    float getInitialPosition() const;
    float getInitialVelocity() const;

    float getPos(float T) const;
    float getVelocity(float T) const;

    void updateExternalForce(float forceFunction, float T);
    void createPlot(float T) const;
};

#endif // MECHANICAL_SYSTEM_H
