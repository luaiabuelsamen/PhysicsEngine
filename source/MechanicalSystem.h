#ifndef MECHANICALSYSTEM_H
#define MECHANICALSYSTEM_H


#include <vector>
#include <Eigen/Dense>


class MechanicalSystem {
public:
    using Vector = Eigen::VectorXf;

    MechanicalSystem(float m, float c, float k, const Vector& x0, const Vector& v0);

    Vector getPos(float t, const Vector& x, const Vector& v, float h);
    Vector getVelocity(float t, const Vector& x, const Vector& v, float h);

    // Getter methods
    float getMass() const;
    float getDamping() const;
    float getSpringConstant() const;
    Vector getInitialPosition() const;
    Vector getInitialVelocity() const;

    // Setter methods
    void setMass(float m);
    void setStiffness(float k);
    void setDamping(float c);

    void updateExternalForce(const Vector& force);
    void setGravity(const Vector& g); // New method

    void createPlot(float T);

    // RK4 and ODE solver
    Vector rk4(Vector (*f)(float, const Vector&, const MechanicalSystem&), float t, const Vector& y, float h, const MechanicalSystem& system);

private:
    float mass, damping, springConstant;
    Vector initialPosition, initialVelocity;
    Vector externalForce;
    Vector gravity; // New member
    Vector positionDerivative(float t, const Vector& x, const Vector& v) const;
    Vector velocityDerivative(float t, const Vector& x, const Vector& v) const;
    static Vector systemOde(float t, const Vector& y, const MechanicalSystem& system);
};

#endif // MECHANICALSYSTEM_H
