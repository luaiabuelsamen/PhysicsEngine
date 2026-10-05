// Tests for phys::ElasticPatch against classical contact mechanics.
// Run through CTest (`ctest`) or directly (`./test_tactile`).

#include <cmath>
#include <cstdio>
#include <vector>

#include "phys/tactile.h"

using namespace phys;

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "  %s:%d: CHECK failed: %s\n", __FILE__,      \
                         __LINE__, #cond);                                     \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

static const double kPi = 3.14159265358979;
static const float E = 3e5f, NU = 0.5f;
static const double E_STAR = E / (1.0 - NU * NU);
static const double G = E / (2.0 * (1.0 + NU));

static ElasticPatchDesc patch(float size, float cell, float mu = 0.5f) {
    ElasticPatchDesc d;
    d.width = d.height = size;
    d.cell = cell;
    d.youngs_modulus = E;
    d.poisson = NU;
    d.friction = mu;
    d.max_iterations = 2000;
    d.tolerance = 1e-11f;
    return d;
}

// Press to depth `depth` in `steps` increments (warm starts).
static void press(ElasticPatch& p, const Indenter& ind, float depth, int steps = 8) {
    for (int s = 1; s <= steps; s++) p.step(ind, {0.0f, 0.0f, -depth * s / steps});
}

// Sphere: F = 4/3 E* sqrt(R) d^1.5, contact radius a = sqrt(R d),
// pressure p0 sqrt(1 - r^2/a^2) with p0 = 3F / (2 pi a^2).
static void test_hertz_sphere() {
    const float R = 5e-3f;
    for (float depth : {1e-4f, 2e-4f, 3e-4f}) {
        ElasticPatch p(patch(5e-3f, 6e-5f));
        Indenter ind;
        ind.radius = R;
        press(p, ind, depth);
        double F = p.force().z;
        double F_exact = 4.0 / 3.0 * E_STAR * std::sqrt(R) * std::pow(depth, 1.5);
        double a = std::sqrt(R * depth);
        double area = p.contact_cells() * 6e-5 * 6e-5;
        double a_sim = std::sqrt(area / kPi);
        // Pressure profile against Hertz, over cells well inside the contact.
        double p0 = 3.0 * F_exact / (2.0 * kPi * a * a), worst = 0.0;
        for (int i = 0; i < p.cells(); i++) {
            double r = std::hypot(p.cell_x(i), p.cell_y(i));
            if (r > 0.8 * a) continue;
            worst = std::fmax(worst, std::fabs(p.pressure(i) - p0 * std::sqrt(1 - r * r / (a * a))) / p0);
        }
        std::printf("  d = %.1f um: F = %.4f N (Hertz %.4f, %+.2f%%), a = %.3f mm (Hertz %.3f), "
                    "max pressure error inside 0.8a: %.2f%% of p0\n",
                    depth * 1e6, F, F_exact, 100 * (F / F_exact - 1), a_sim * 1e3, a * 1e3, 100 * worst);
        CHECK(std::fabs(F / F_exact - 1) < 0.02);
        CHECK(std::fabs(a_sim / a - 1) < 0.03);
        CHECK(worst < 0.03);
    }
}

// Flat cylindrical punch: F = 2 a E* d (linear in depth).
static void test_flat_punch() {
    const float a = 1e-3f;
    for (float depth : {5e-5f, 1e-4f}) {
        ElasticPatch p(patch(4e-3f, 5e-5f));
        Indenter ind;
        ind.kind = Indenter::FlatPunch;
        ind.radius = a;
        press(p, ind, depth, 2);
        double F = p.force().z, F_exact = 2.0 * a * E_STAR * depth;
        std::printf("  d = %.0f um: F = %.4f N (Boussinesq %.4f, %+.2f%%)\n", depth * 1e6, F, F_exact,
                    100 * (F / F_exact - 1));
        CHECK(std::fabs(F / F_exact - 1) < 0.04);
    }
}

// Cattaneo-Mindlin: a sphere pressed with load P then displaced sideways by u
// carries Q with  u = 3 mu P (2 - nu) / (16 G a) [1 - (1 - Q / mu P)^(2/3)],
// sticking inside c = a (1 - Q / mu P)^(1/3); full sliding at Q = mu P.
static void test_cattaneo_mindlin() {
    const float R = 5e-3f, depth = 2e-4f, mu = 0.5f, cell = 6e-5f;
    ElasticPatch p(patch(4e-3f, cell, mu));
    Indenter ind;
    ind.radius = R;
    press(p, ind, depth);
    double P = p.force().z;
    double a = std::sqrt(R * depth);
    double u_star = 3.0 * mu * P * (2.0 - NU) / (16.0 * G * a);  // displacement at full slip

    double worst_u = 0.0, worst_c = 0.0;
    const int steps = 60;
    for (int s = 1; s <= steps; s++) {
        double u = 1.3 * u_star * s / steps;
        p.step(ind, {(float)u, 0.0f, -depth});
        double ratio = p.force().x / (mu * P);
        if (ratio < 0.95 && ratio > 0.1) {
            double u_exact = u_star * (1.0 - std::pow(1.0 - ratio, 2.0 / 3.0));
            worst_u = std::fmax(worst_u, std::fabs(u - u_exact) / u_star);
            // Stick region: radius of the area of sticking cells.
            int stick = 0;
            for (int i = 0; i < p.cells(); i++) stick += p.pressure(i) > 0 && p.sticking(i);
            double c_sim = std::sqrt(stick * cell * cell / kPi);
            double c_exact = a * std::cbrt(1.0 - ratio);
            worst_c = std::fmax(worst_c, std::fabs(c_sim - c_exact) / a);
        }
    }
    double final_ratio = p.force().x / (mu * P);
    std::printf("  P = %.4f N, slip onset u* = %.1f um; max |u - u_CM| = %.2f%% of u*, "
                "max stick-radius error %.2f%% of a; final Q/(mu P) = %.4f\n",
                P, u_star * 1e6, 100 * worst_u, 100 * worst_c, final_ratio);
    CHECK(worst_u < 0.05);
    CHECK(worst_c < 0.06);
    CHECK(std::fabs(final_ratio - 1.0) < 0.01);
    CHECK(std::fabs(p.force().y) < 1e-6 * P);
}

// Reversing the slide unloads along a different path (hysteresis), and the
// contact keeps a memory of the slide: shear does not return to zero when
// the tip returns to its starting position.
static void test_hysteresis() {
    const float R = 5e-3f, depth = 2e-4f, mu = 0.5f;
    ElasticPatch p(patch(4e-3f, 8e-5f, mu));
    Indenter ind;
    ind.radius = R;
    press(p, ind, depth);
    double P = p.force().z;
    double a = std::sqrt(R * depth);
    double u_star = 3.0 * mu * P * (2.0 - NU) / (16.0 * G * a);
    double u_max = 0.6 * u_star;
    std::vector<double> up, down;
    for (int s = 0; s <= 30; s++) {
        p.step(ind, {(float)(u_max * s / 30), 0.0f, -depth});
        up.push_back(p.force().x);
    }
    for (int s = 30; s >= 0; s--) {
        p.step(ind, {(float)(u_max * s / 30), 0.0f, -depth});
        down.push_back(p.force().x);
    }
    // Masing / Mindlin-Deresiewicz: unloading from Q* follows
    // Q(u) = Q* - 2 Q_load((u_max - u) / 2).
    auto Q_load = [&](double u) {
        double x = std::fmin(u / u_star, 1.0);
        return mu * P * (1.0 - std::pow(1.0 - x, 1.5));
    };
    double worst = 0.0;
    for (int s = 0; s <= 30; s++) {
        double u = u_max * (30 - s) / 30;
        double expected = up.back() - 2.0 * Q_load((u_max - u) / 2.0);
        worst = std::fmax(worst, std::fabs(down[s] - expected) / (mu * P));
    }
    std::printf("  residual shear back at u = 0: %.4f N (%.1f%% of mu P); "
                "max deviation from Masing unloading: %.2f%% of mu P\n",
                down.back(), 100 * down.back() / (mu * P), 100 * worst);
    CHECK(down.back() < -0.02 * mu * P);  // memory: shear reversed, not zero
    CHECK(worst < 0.04);
}

// Winkler mode: independent springs give F = pi k R d^2 for a sphere, and
// the brush shear curve Q / (mu P) = 1 - (1 - u / u*)^2 with u* = mu k d / k_t.
static void test_winkler() {
    const float R = 5e-3f, depth = 2e-4f, mu = 0.5f, k = 2e9f, ratio = 0.4f, cell = 5e-5f;
    ElasticPatchDesc d = patch(4e-3f, cell, mu);
    d.model = ElasticPatchDesc::Winkler;
    d.winkler_stiffness = k;
    d.winkler_shear_ratio = ratio;
    ElasticPatch p(d);
    Indenter ind;
    ind.radius = R;
    press(p, ind, depth);
    double P = p.force().z, P_exact = kPi * k * R * depth * depth;
    double u_star = mu * depth / ratio;  // the centre cell (pressure k d) slips last
    double worst = 0.0;
    for (int s = 1; s <= 40; s++) {
        double u = 1.2 * u_star * s / 40;
        p.step(ind, {(float)u, 0.0f, -depth});
        double x = std::fmin(u / u_star, 1.0);
        double expected = 1.0 - (1.0 - x) * (1.0 - x);
        worst = std::fmax(worst, std::fabs(p.force().x / (mu * P) - expected));
    }
    std::printf("  F = %.4f N (pi k R d^2 = %.4f, %+.2f%%); brush curve max error %.2f%% of mu P\n", P, P_exact,
                100 * (P / P_exact - 1), 100 * worst);
    CHECK(std::fabs(P / P_exact - 1) < 0.02);  // grid quadrature of the contact edge
    CHECK(worst < 0.02);
}

// A compliant holder in series: tip displacement = Q / k_s + contact
// displacement (Cattaneo-Mindlin).
static void test_series_holder() {
    const float R = 5e-3f, depth = 2e-4f, mu = 0.5f, cell = 6e-5f;
    ElasticPatchDesc d = patch(4e-3f, cell, mu);
    ElasticPatch rigid(d);
    Indenter ind;
    ind.radius = R;
    press(rigid, ind, depth);
    double P = rigid.force().z, a = std::sqrt(R * depth);
    double u_star = 3.0 * mu * P * (2.0 - NU) / (16.0 * G * a);
    const float ks = (float)(0.5 * mu * P / u_star);  // holder about as compliant as the contact
    d.series_tangential_stiffness = ks;
    ElasticPatch p(d);
    press(p, ind, depth);
    double worst = 0.0;
    const double u_total = 3.0 * u_star + mu * P / ks;
    for (int s = 1; s <= 60; s++) {
        double u = u_total * s / 60;
        p.step(ind, {(float)u, 0.0f, -depth});
        double Q = p.force().x, ratio = Q / (mu * P);
        if (ratio > 0.05 && ratio < 0.95) {
            double u_exact = Q / ks + u_star * (1.0 - std::pow(1.0 - ratio, 2.0 / 3.0));
            worst = std::fmax(worst, std::fabs(u - u_exact) / u_star);
        }
    }
    std::printf("  k_s = %.1f N/m: max |u - (Q/k_s + u_CM)| = %.2f%% of u*; final Q/(mu P) = %.4f\n", ks,
                100 * worst, p.force().x / (mu * P));
    CHECK(worst < 0.05);
    CHECK(std::fabs(p.force().x / (mu * P) - 1.0) < 0.01);
}

int main() {
    struct Test { const char* name; void (*fn)(); };
    const Test tests[] = {
        {"hertz_sphere", test_hertz_sphere},
        {"flat_punch", test_flat_punch},
        {"cattaneo_mindlin", test_cattaneo_mindlin},
        {"hysteresis", test_hysteresis},
        {"winkler", test_winkler},
        {"series_holder", test_series_holder},
    };
    for (const Test& t : tests) {
        int before = g_failures;
        std::printf("%s\n", t.name);
        t.fn();
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    std::printf("\n%s (%d failed checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
