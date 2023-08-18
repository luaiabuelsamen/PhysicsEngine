#ifndef RK4_SOLVER_H
#define RK4_SOLVER_H

#include <cmath>
#include <functional>

template<typename Func, typename... Args>
float rk4(float x0, float y0, float xn, float h, Func&& f, Args&&... args) {
    float yn, k1, k2, k3, k4, k;

    /* Runge Kutta Method */
    while (x0 < xn)
    {
        k1 = h * f(x0, y0, std::forward<Args>(args)...);
        k2 = h * f(x0 + h / 2, y0 + k1 / 2, std::forward<Args>(args)...);
        k3 = h * f(x0 + h / 2, y0 + k2 / 2, std::forward<Args>(args)...);
        k4 = h * f(x0 + h, y0 + k3, std::forward<Args>(args)...);
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        yn = y0 + k;
        x0 = x0 + h;
        y0 = yn;
    }

    return yn;
}

#endif // RK4_SOLVER_H
