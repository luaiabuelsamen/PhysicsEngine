#ifndef SPRINGWIDGET_H
#define SPRINGWIDGET_H

#include <QWidget>
#include <QTimer>
#include "MechanicalSystem.h"

class SpringWidget : public QWidget {
    Q_OBJECT  // Add this macro

public:
    SpringWidget(QWidget *parent = nullptr);
    void startSimulation();

protected:
    void paintEvent(QPaintEvent *event) override;

private slots:
    void updateSimulation();

private:
    QTimer *timer;
    MechanicalSystem *system;
    float t;
    float x, v, h;
};

#endif // SPRINGWIDGET_H
