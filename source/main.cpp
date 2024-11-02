#include <QApplication>
#include "SpringWidget.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    
    SpringWidget w;
    w.resize(800, 600);
    w.show();
    w.startSimulation();
    
    return a.exec();
}
