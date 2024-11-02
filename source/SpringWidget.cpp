#include "SpringWidget.h"
#include <QPainter>

SpringWidget::SpringWidget(QWidget *parent) 
    : QWidget(parent), t(0), h(0.1), x(0), v(0) {
    
    system = new MechanicalSystem(1.0, 0.1, 5.0, 10.0, 0.0);  // Mass, damping, spring constant, initial pos, initial vel
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &SpringWidget::updateSimulation);
}

void SpringWidget::startSimulation() {
    timer->start(20); // Update every 20 ms
}

void SpringWidget::updateSimulation() {
    // Update position and velocity
    x = system->getPos(t, x, v, h);
    v = system->getVelocity(t, x, v, h);
    t += h;
    update(); // Trigger repaint
}

void SpringWidget::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw fixed point
    int fixedX = width() / 2;
    int fixedY = height() / 2;
    painter.setBrush(Qt::black);
    painter.drawEllipse(fixedX - 5, fixedY - 5, 10, 10);

    // Draw the spring and mass
    int massX = static_cast<int>(x * 50) + fixedX;
    painter.setBrush(Qt::red);
    painter.drawEllipse(massX - 10, fixedY - 10, 20, 20);
    painter.setPen(Qt::black);
    painter.drawLine(fixedX, fixedY, massX, fixedY);
}
