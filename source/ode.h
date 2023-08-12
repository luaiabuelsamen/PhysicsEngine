#ifndef RK4_SOLVER_H
#define RK4_SOLVER_H

#include <cmath>
template<typename Func>
float rk4(float x0, float y0, float xn, int n, Func&& f);

#endif // RK4_SOLVER_H
