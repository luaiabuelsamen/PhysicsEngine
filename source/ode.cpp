#include "ode.h"

using namespace std;

template<typename Func>
float rk4(float x0, float y0, float xn, int n, Func&& f) {
    float h, yn, k1, k2, k3, k4, k;
    int i;

    /* Calculating step size (h) */
    h = (xn - x0) / n;

    /* Runge Kutta Method */
    for (i = 0; i < n; i++)
    {
        k1 = h * f(x0, y0);
        k2 = h * f(x0 + h / 2, y0 + k1 / 2);
        k3 = h * f(x0 + h / 2, y0 + k2 / 2);
        k4 = h * f(x0 + h, y0 + k3);
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        yn = y0 + k;
        x0 = x0 + h;
        y0 = yn;
    }

    return yn;
}
