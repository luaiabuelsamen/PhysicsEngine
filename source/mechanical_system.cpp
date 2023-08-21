#include "ode.h"
#include <iostream>
#include <cmath>
#include <functional>
#include "mechanical_system.h"
#include <matplotlibcpp.h>
#include <opencv2/opencv.hpp>
namespace plt = matplotlibcpp;

using namespace std;
using namespace placeholders;
using namespace cv;

MechanicalSystem::MechanicalSystem(float m, float c, float k, float F_ext, float x0, float v0)
    : mass(m), damping(c), springConstant(k), externalForce(), initialPosition(x0), initialVelocity(v0) {}

// Getter methods
float MechanicalSystem::getMass() const { return mass; }
float MechanicalSystem::getDamping() const { return damping; }
float MechanicalSystem::getSpringConstant() const { return springConstant; }
float MechanicalSystem::getInitialPosition() const { return initialPosition; }
float MechanicalSystem::getInitialVelocity() const { return initialVelocity; }


float MechanicalSystem::positionDerivative(float t, float x, float v) const {
    return v;
}

float MechanicalSystem::velocityDerivative(float t, float x, float v) const {
    // Calculate acceleration based on damping, spring force, mass, and external force
    float acceleration = (- getDamping() * v - getSpringConstant() * x) / getMass();
    return acceleration;
}

float MechanicalSystem::getVelocity(float t, float x, float v, float h) const {
    std::vector<double> y0 = rk4(systemOde, t, {x, v}, h, *this);
    return y0[1];
}

float MechanicalSystem::getPos(float t, float x, float v, float h) const {
    std::vector<double> y0 = rk4(systemOde, t,  {x, v}, h, *this);
    return y0[0];
}

// Method to update external force using a given force function
void MechanicalSystem::updateExternalForce(float forceFunction, float T) {
    externalForce = forceFunction;
}

std::vector<double> MechanicalSystem::systemOde(float t, std::vector<double> y, const MechanicalSystem& system) {
    return {system.positionDerivative(t, y[0], y[1]), system.velocityDerivative(t, y[0], y[1])};
}

std::vector<double> MechanicalSystem::rk4(std::vector<double> (*f)(float, std::vector<double>, const MechanicalSystem&), float t, const std::vector<double>& y, float h, const MechanicalSystem& system) {
    std::vector<double> k1 = f(t, y, system);

    std::vector<double> k1_scaled;
    k1_scaled.reserve(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        k1_scaled.push_back(k1[i] * h/2 + y[i]);
    }
    std::vector<double> k2 = f(t + h/2, k1_scaled, system);

    std::vector<double> k2_scaled;
    k2_scaled.reserve(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        k2_scaled.push_back(k2[i] * h/2 + y[i]);
    }
    std::vector<double> k3 = f(t + h/2, k2_scaled, system);
    
    std::vector<double> k3_scaled;
    k3_scaled.reserve(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        k3_scaled.push_back(k3[i] * h + y[i]);
    }
    std::vector<double> k4 = f(t + h, k3_scaled, system);
    
    std::vector<double> y1;
    for (size_t i = 0; i < y.size(); ++i) {
        y1.push_back(y[i] + h/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]));
    }

    return y1;
};

void MechanicalSystem::createPlot(float T) {

    // Initial conditions
    float t = 0.0;  // Initial time
    float y = initialPosition;  // Initial position
    float v = initialVelocity;
    float h = 0.1;  // Step size

    // Create a video writer
    int frameWidth = 800;
    int frameHeight = 600;

    std::string videoFilename = "spring_motion_video.avi";
    VideoWriter video(videoFilename, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, Size(frameWidth, frameHeight));

    std::vector<float> timePoints;
    std::vector<float> positionPoints;
    std::vector<float> velocityPoints;
    while (t <= T) {  // Adjust the end time as needed
        Mat frame(frameHeight, frameWidth, CV_8UC3, Scalar(255, 255, 255));

        // Calculate position and velocity at current time
        float currentPos = getPos(t, y, v, h);
        float currentVel = getVelocity(t, y, v, h);

        // Draw the mass as a circle
        int massX = static_cast<int>(currentPos * 50) + frameWidth / 2;
        int massY = frameHeight / 2;
        circle(frame, Point(massX, massY), 10, Scalar(0, 0, 255), -1);

        // Draw a fixed point at 0
        int fixedPointX = frameWidth / 2;
        int fixedPointY = frameHeight / 2;
        circle(frame, Point(fixedPointX, fixedPointY), 5, Scalar(0, 0, 0), -1);

        // Draw the spring connecting the mass and the fixed point
        int springStartX = fixedPointX;
        int springStartY = fixedPointY;
        int springEndX = massX;
        int springEndY = massY;
        line(frame, Point(springStartX, springStartY), Point(springEndX, springEndY), Scalar(0, 0, 0), 2);

        // Write the frame to the video
        video.write(frame);

        // Update time and initial conditions
        t += h;
        y = currentPos;
        v = currentVel;
        positionPoints.push_back(y);
        velocityPoints.push_back(v);
        timePoints.push_back(t);
    }

    // Create a plot
    plt::plot(timePoints, positionPoints, {{"label", "Position"}});
    plt::plot(timePoints, velocityPoints, {{"label", "Velocity"}});
    plt::xlabel("Time");
    plt::ylabel("Value");
    plt::title("Trajectory of the Mass");

    // Add legend
    plt::legend();
    plt::save("spring_figure.png");


    // Release the video writer
    video.release();
}
