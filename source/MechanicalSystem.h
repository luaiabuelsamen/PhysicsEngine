#ifndef MECHANICALSYSTEM_H
#define MECHANICALSYSTEM_H

#include <vector>

class MechanicalSystem {
public:
    MechanicalSystem(float m, float c, float k, float x0, float v0);
    float getPos(float t, float x, float v, float h);
    float getVelocity(float t, float x, float v, float h);

    // Getter methods
    float getMass() const;
    float getDamping() const;
    float getSpringConstant() const;
    float getInitialPosition() const;
    float getInitialVelocity() const;

    // Setter methods
    void setMass(float m);
    void setStiffness(float k);
    void setDamping(float c);

    void updateExternalForce(float force);

    void createPlot(float T);

    // RK4 and ODE solver
    std::vector<double> rk4(std::vector<double> (*f)(float, std::vector<double>, const MechanicalSystem&), 
                            float t, const std::vector<double>& y, float h, const MechanicalSystem& system);

private:
    float mass, damping, springConstant, initialPosition, initialVelocity;
    float positionDerivative(float t, float x, float v) const;
    float velocityDerivative(float t, float x, float v) const;
    static std::vector<double> systemOde(float t, std::vector<double> y, const MechanicalSystem& system);
};

#endif // MECHANICALSYSTEM_H
