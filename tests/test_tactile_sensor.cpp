// Tests for tactile sensor pads in World (phys::TactileSensorDesc): the
// pressure and shear they report against closed-form contact mechanics, and
// CPU / CUDA agreement.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "phys/phys.h"

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

namespace {

const float kPi = 3.14159265f;
const int C = World::kTactileChannels;
enum { kP = 0, kQx = 1, kQy = 2, kW = 3, kVx = 4, kVy = 5, kStick = 6 };

// A pad (the bottom face of a box) pressed onto a fixed sphere: the box
// hangs from a carriage on a vertical slider (force = normal load) and
// moves on a horizontal slider relative to it (force = shear load). No
// gravity, so the loads are exactly the slider forces.
struct Rig {
    float R = 0.01f;         // sphere radius
    float E = 3e5f, nu = 0.5f;
    float mu = 0.5f;
    float pad = 0.012f;      // pad side
    int cells = 32;
    ModelDesc model;
    int pad_body = 1;

    Rig() {
        model.gravity = {0.0f, 0.0f, 0.0f};
        // Stiff rigid contact pressed through a joint chatters at 10 substeps
        // (the pad force varies ~8% step to step under shear, and static
        // friction creeps near the cone); 40 substeps hold it steady.
        model.substeps = 40;
        BodyDesc carriage = BodyDesc::none(0.5f, {1e-3f, 1e-3f, 1e-3f});
        BodyDesc box = BodyDesc::box({0.02f, 0.02f, 0.005f}, 0.1f);
        BodyDesc ball = BodyDesc::sphere(R, 0.0f);
        box.friction = ball.friction = mu;
        box.restitution = ball.restitution = 0.0f;
        model.bodies = {carriage, box, ball};  // 0, 1, 2
        JointDesc lift = JointDesc::slider(-1, 0, {0, 0, R + 0.005f}, {0, 0, 0}, {0, 0, 1});
        lift.actuator = Actuator::Torque;
        lift.damping = 2.0f;
        JointDesc slide = JointDesc::slider(0, 1, {0, 0, 0}, {0, 0, 0}, {1, 0, 0});
        slide.actuator = Actuator::Torque;
        slide.damping = 2.0f;
        model.joints = {lift, slide};

        TactileSensorDesc t;
        t.body = pad_body;
        t.origin = {0.0f, 0.0f, -0.005f};
        t.frame = {0.0f, 1.0f, 0.0f, 0.0f};  // half turn about x: +z points down, x stays x
        t.width = t.height = pad;
        t.nx = t.ny = cells;
        t.youngs_modulus = E;
        t.poisson = nu;
        model.tactile_sensors = {t};
    }

    // Press env e with normal load W[e] and shear Q[e] for `seconds`.
    World run(Device dev, const std::vector<float>& W, const std::vector<float>& Q, float seconds) const {
        int nenv = (int)W.size();
        World w(model, nenv, dev);
        HostState s(nenv * 3);
        for (int e = 0; e < nenv; e++) {
            s.pz[e * 3 + 0] = R + 0.005f;
            s.pz[e * 3 + 1] = R + 0.005f;
        }
        w.set_state(s);
        // Press first; shear once the gel has taken the load.
        std::vector<float> ctrl(nenv * 2);
        for (int e = 0; e < nenv; e++) ctrl[e * 2 + 0] = -W[e];
        w.set_controls(ctrl);
        w.step(1.0f / 240.0f, 60);
        for (int e = 0; e < nenv; e++) ctrl[e * 2 + 1] = Q[e];
        w.set_controls(ctrl);
        w.step(1.0f / 240.0f, (int)(seconds * 240.0f));
        return w;
    }
};

struct Reading {
    std::vector<float> cells, force;
    int n;
    float cell_area;
    const float* env(int e, int ch) const { return &cells[((size_t)e * n) * C + (size_t)ch * n]; }
};

Reading read(World& w, const Rig& rig) {
    Reading r;
    w.get_tactile(r.cells, r.force);
    r.n = rig.cells * rig.cells;
    float c = rig.pad / rig.cells;
    r.cell_area = c * c;
    return r;
}

// Radius of a circle with the area of the cells where `mask` holds.
float equivalent_radius(int count, float cell_area) { return std::sqrt(count * cell_area / kPi); }

void test_hertz() {
    Rig rig;
    std::vector<float> W = {0.5f, 1.0f, 2.0f};
    World w = rig.run(Device::CUDA, W, {0.0f, 0.0f, 0.0f}, 1.0f);
    Reading r = read(w, rig);
    float e_star = rig.E / (1.0f - rig.nu * rig.nu);
    for (int e = 0; e < (int)W.size(); e++) {
        const float* p = r.env(e, kP);
        const float* defl = r.env(e, kW);
        int contact = 0;
        float total = 0.0f, peak = 0.0f, centre_defl = 0.0f;
        for (int i = 0; i < r.n; i++) {
            contact += p[i] > 0.0f;
            total += p[i] * r.cell_area;
            peak = std::fmax(peak, p[i]);
            centre_defl = std::fmax(centre_defl, defl[i]);
        }
        float a = std::cbrt(3.0f * W[e] * rig.R / (4.0f * e_star));
        float p0 = 3.0f * W[e] / (2.0f * kPi * a * a);
        float delta = a * a / rig.R;
        float a_sim = equivalent_radius(contact, r.cell_area);
        std::printf("  W = %.1f N: reported %.4f N, sum p A %.4f N | contact radius %.3f mm (Hertz %.3f) | "
                    "peak %.1f kPa (Hertz %.1f) | indentation %.3f mm (Hertz %.3f)\n",
                    W[e], r.force[e * 3 + 2], total, a_sim * 1e3f, a * 1e3f, peak * 1e-3f, p0 * 1e-3f,
                    centre_defl * 1e3f, delta * 1e3f);
        CHECK(std::fabs(r.force[e * 3 + 2] - W[e]) < 0.01f * W[e]);
        CHECK(std::fabs(total - r.force[e * 3 + 2]) < 1e-4f * W[e]);
        float cell = rig.pad / rig.cells;
        CHECK(std::fabs(a_sim - a) < std::fmax(0.03f * a, 0.5f * cell));
        CHECK(std::fabs(peak - p0) < 0.05f * p0);
        CHECK(std::fabs(centre_defl - delta) < 0.05f * delta);
    }
}

// Mindlin: under shear Q < mu W at constant W, a central disc of radius
// c = a (1 - Q / (mu W))^(1/3) sticks and the annulus around it slips. The
// rig's loads are checked against the applied ones; the stick zone against
// Mindlin at the loads the pad reports.
void test_partial_slip() {
    Rig rig;
    std::vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> Q = {0.0f, 0.15f, 0.3f, 0.4f};  // 0, 0.3, 0.6, 0.8 of mu W
    World w = rig.run(Device::CUDA, W, Q, 1.0f);
    Reading r = read(w, rig);
    float e_star = rig.E / (1.0f - rig.nu * rig.nu);
    float cell = rig.pad / rig.cells;
    for (int e = 0; e < (int)W.size(); e++) {
        int stick = 0;
        float qx = 0.0f, qy = 0.0f, worst_cone = 0.0f;
        for (int i = 0; i < r.n; i++) {
            float p = r.env(e, kP)[i], sx = r.env(e, kQx)[i], sy = r.env(e, kQy)[i];
            stick += r.env(e, kStick)[i] > 0.5f;
            qx += sx * r.cell_area;
            qy += sy * r.cell_area;
            if (p > 0.0f) worst_cone = std::fmax(worst_cone, std::hypot(sx, sy) / (rig.mu * p));
        }
        float Wr = r.force[e * 3 + 2], Qr = std::hypot(r.force[e * 3 + 0], r.force[e * 3 + 1]);
        float a = std::cbrt(3.0f * Wr * rig.R / (4.0f * e_star));
        float c = a * std::cbrt(1.0f - Qr / (rig.mu * Wr));
        float c_sim = equivalent_radius(stick, r.cell_area);
        std::printf("  applied Q / mu W = %.1f: pad reports W %.4f N, shear (%.4f, %.4f) N, sum q A (%.4f, %.4f) "
                    "| stick radius %.3f mm (Mindlin at the reported loads %.3f) | max |q| / mu p = %.3f\n",
                    Q[e] / (rig.mu * W[e]), Wr, r.force[e * 3 + 0], r.force[e * 3 + 1], qx, qy, c_sim * 1e3f,
                    c * 1e3f, worst_cone);
        // The pad is pushed +x and held by friction, so the sphere pushes it -x.
        CHECK(std::fabs(Wr - W[e]) < 0.02f * W[e]);
        CHECK(std::fabs(r.force[e * 3 + 0] + Q[e]) < 0.03f * W[e]);
        CHECK(std::fabs(r.force[e * 3 + 1]) < 1e-3f * W[e]);
        CHECK(std::fabs(qx - r.force[e * 3 + 0]) < 1e-4f * W[e]);
        CHECK(worst_cone <= 1.0f + 1e-3f);
        CHECK(std::fabs(c_sim - c) < std::fmax(0.04f * a, 0.75f * cell));
    }
}

// Beyond mu W the pad slides: every cell slips, with shear proportional to
// pressure, |q| = (|Q| / W) p, and |Q| close to mu W.
void test_full_slip() {
    Rig rig;
    World w = rig.run(Device::CUDA, {1.0f}, {0.8f}, 0.08f);  // mid-slide, still on the sphere
    Reading r = read(w, rig);
    int stick = 0, contact = 0;
    float Wr = r.force[2], Qr = std::hypot(r.force[0], r.force[1]), worst = 0.0f;
    for (int i = 0; i < r.n; i++) {
        float p = r.env(0, kP)[i];
        stick += r.env(0, kStick)[i] > 0.5f;
        if (p > 0.0f) {
            contact++;
            float q = std::hypot(r.env(0, kQx)[i], r.env(0, kQy)[i]);
            worst = std::fmax(worst, std::fabs(q - Qr / Wr * p) / (Qr / Wr * p));
        }
    }
    std::printf("  sliding: shear %.4f N, mu W %.4f N, %d / %d cells stick, max | |q| - (Q / W) p | / ((Q / W) p) "
                "= %.2g\n", Qr, rig.mu * Wr, stick, contact, worst);
    CHECK(std::fabs(Qr - rig.mu * Wr) < 0.05f * Wr);
    CHECK(stick == 0);
    CHECK(worst < 1e-3f);
}

// The CPU and CUDA backends produce the same readings bit for bit.
void test_backends_agree() {
    Rig rig;
    rig.cells = 16;
    rig.model.tactile_sensors[0].nx = rig.model.tactile_sensors[0].ny = 16;
    std::vector<float> W = {0.5f, 1.0f, 1.5f}, Q = {0.1f, 0.2f, 1.0f};
    World a = rig.run(Device::CPU, W, Q, 0.25f);
    World b = rig.run(Device::CUDA, W, Q, 0.25f);
    std::vector<float> ca, fa, cb, fb;
    a.get_tactile(ca, fa);
    b.get_tactile(cb, fb);
    bool same = ca.size() == cb.size() && fa.size() == fb.size() &&
                std::memcmp(ca.data(), cb.data(), ca.size() * sizeof(float)) == 0 &&
                std::memcmp(fa.data(), fb.data(), fa.size() * sizeof(float)) == 0;
    double nonzero = 0;
    for (float v : ca) nonzero += v != 0.0f;
    std::printf("  CPU vs CUDA: %s (%zu values, %.0f non-zero)\n", same ? "bit-identical" : "DIFFERENT",
                ca.size(), nonzero);
    CHECK(same);
    CHECK(nonzero > 100);
}

// A flat pad on a flat floor: the whole pad touches, and the pressure
// rises towards the edges (flat-punch-like).
void test_flat_on_plane() {
    ModelDesc m;
    m.gravity = {0.0f, -9.81f, 0.0f};
    BodyDesc floor = BodyDesc::plane();
    BodyDesc box = BodyDesc::box({0.01f, 0.005f, 0.01f}, 0.2f);
    m.bodies = {floor, box};
    TactileSensorDesc t;
    t.body = 1;
    t.origin = {0.0f, -0.005f, 0.0f};
    t.frame = {0.70710678f, 0.70710678f, 0.0f, 0.0f};  // quarter turn about x: +z -> -y
    t.width = t.height = 0.02f;
    t.nx = t.ny = 20;
    m.tactile_sensors = {t};
    World w(m, 1, Device::CUDA);
    HostState s(2);
    s.py[1] = 0.005f;
    w.set_state(s);
    w.step(1.0f / 240.0f, 240);
    std::vector<float> cells, force;
    w.get_tactile(cells, force);
    int n = 400, contact = 0;
    float total = 0.0f, area = 1e-6f;
    for (int i = 0; i < n; i++) {
        contact += cells[i] > 0.0f;
        total += cells[i] * area;
    }
    float centre = cells[10 * 20 + 10], edge = cells[10 * 20 + 0];
    std::printf("  box on floor: normal %.4f N (weight %.4f), %d / %d cells touch, edge / centre "
                "pressure %.2f\n", force[2], 0.2f * 9.81f, contact, n, edge / centre);
    CHECK(std::fabs(force[2] - 0.2f * 9.81f) < 0.02f * 0.2f * 9.81f);
    CHECK(std::fabs(total - force[2]) < 1e-3f * force[2]);
    CHECK(contact == n);
    CHECK(edge > 1.5f * centre);
}


// Two-way coupling: the pad body itself indents the sphere by Hertz's
// approach, and under shear it is displaced along Mindlin's curve
// u = u* (1 - (1 - Q / mu W)^(2/3)), u* = 3 mu W / (2 k_t), with
// k_t = 2 E* a / tangential_ratio the initial tangential stiffness.
void test_coupled_indentation_and_mindlin() {
    Rig rig;
    std::vector<float> W = {1.0f, 1.0f, 1.0f, 2.0f};
    std::vector<float> Q = {0.0f, 0.15f, 0.3f, 0.0f};
    World w = rig.run(Device::CUDA, W, Q, 1.0f);
    HostState s;
    w.get_state(s);
    float e_star = rig.E / (1.0f - rig.nu * rig.nu);
    float ratio = (2.0f - rig.nu) / (2.0f * (1.0f - rig.nu));
    for (int e = 0; e < (int)W.size(); e++) {
        float pad_z = s.pz[e * 3 + 1] - 0.005f;
        float indent = rig.R - pad_z;  // sphere top minus pad surface
        float a = std::cbrt(3.0f * W[e] * rig.R / (4.0f * e_star));
        float delta = a * a / rig.R;
        float kt = 2.0f * e_star * a / ratio;
        float ustar = 1.5f * rig.mu * W[e] / kt;
        float u = ustar * (1.0f - std::pow(1.0f - Q[e] / (rig.mu * W[e]), 2.0f / 3.0f));
        float ux = s.px[e * 3 + 1];
        std::printf("  W %.1f N, Q %.2f N: pad indents %.3f mm (Hertz %.3f); shear displacement %.3f mm "
                    "(Mindlin %.3f)\n", W[e], Q[e], indent * 1e3f, delta * 1e3f, ux * 1e3f, u * 1e3f);
        CHECK(std::fabs(indent - delta) < 0.03f * delta);
        CHECK(std::fabs(ux - u) < 0.08f * u + 5e-6f);
    }
}

// The coupled pad's force is steady from step to step, even at 10 substeps
// (a rigid contact pushed through joints scatters by ~8% there).
void test_coupled_steady() {
    Rig rig;
    rig.model.substeps = 10;
    World w = rig.run(Device::CUDA, {1.0f}, {0.3f}, 1.0f);
    std::vector<float> cells, force;
    double sw = 0, sw2 = 0, sq = 0, sq2 = 0;
    const int n = 120;
    for (int k = 0; k < n; k++) {
        w.step(1.0f / 240.0f);
        w.get_tactile(cells, force);
        sw += force[2];
        sw2 += force[2] * force[2];
        sq += force[0];
        sq2 += force[0] * force[0];
    }
    double mw = sw / n, mq = sq / n;
    double dw = std::sqrt(std::fmax(sw2 / n - mw * mw, 0.0)), dq = std::sqrt(std::fmax(sq2 / n - mq * mq, 0.0));
    std::printf("  10 substeps, over 0.5 s: W %.5f +- %.1e N, Q %.5f +- %.1e N\n", mw, dw, mq, dq);
    CHECK(std::fabs(mw - 1.0) < 0.005);
    CHECK(std::fabs(mq + 0.3) < 0.005);
    CHECK(dw < 1e-3 && dq < 1e-3);
}

// A sharp corner under a coupled pad: the linear gel would let it sink
// without limit (a point load); the gel's thickness caps the indentation at
// half of it, and the pad keeps reporting the load.
void test_coupled_corner() {
    ModelDesc m;
    m.gravity = {0.0f, 0.0f, 0.0f};
    m.substeps = 40;
    BodyDesc carriage = BodyDesc::none(0.5f, {1e-3f, 1e-3f, 1e-3f});
    BodyDesc finger = BodyDesc::box({0.008f, 0.008f, 0.004f}, 0.05f);
    BodyDesc cube = BodyDesc::box({0.005f, 0.005f, 0.005f}, 0.0f);
    finger.restitution = cube.restitution = 0.0f;
    m.bodies = {carriage, finger, cube};
    JointDesc lift = JointDesc::slider(-1, 0, {0, 0, 0.004f}, {0, 0, 0}, {0, 0, 1});
    lift.actuator = Actuator::Torque;
    lift.damping = 2.0f;
    m.joints = {lift, JointDesc::fixed(0, 1, {0, 0, 0}, {0, 0, 0})};
    TactileSensorDesc t;
    t.body = 1;
    t.origin = {0.0f, 0.0f, -0.004f};
    t.frame = {0.0f, 1.0f, 0.0f, 0.0f};
    t.width = t.height = 0.016f;
    t.nx = t.ny = 32;
    m.tactile_sensors = {t};
    World w(m, 1, Device::CUDA);
    HostState s(3);
    s.pz[0] = s.pz[1] = 0.004f;
    // Corner up: turn (1, 1, 1) onto +z.
    float angle = std::acos(1.0f / std::sqrt(3.0f)), sn = std::sin(angle / 2.0f);
    s.pz[2] = -0.005f * std::sqrt(3.0f);
    s.qw[2] = std::cos(angle / 2.0f);
    s.qx[2] = sn / std::sqrt(2.0f);
    s.qy[2] = -sn / std::sqrt(2.0f);
    w.set_state(s);
    w.set_controls({-3.0f, 0.0f});
    w.step(1.0f / 240.0f, 240);
    std::vector<float> cells, force;
    w.get_tactile(cells, force);
    HostState out;
    w.get_state(out);
    float indent = -(out.pz[1] - 0.004f);
    std::printf("  corner at 3 N: pad reports %.4f N, indentation %.2f mm (limit %.2f mm)\n", force[2],
                indent * 1e3f, 0.5f * t.thickness * 1e3f);
    CHECK(std::fabs(force[2] - 3.0f) < 0.03f);
    CHECK(indent > 0.0f && indent < 0.5f * t.thickness * 1.05f);
}

// With coupling off the pad only observes the rigid contact (which, pushed
// through joints, chatters by a few percent step to step).
void test_uncoupled() {
    Rig rig;
    rig.model.tactile_sensors[0].coupled = false;
    World w = rig.run(Device::CUDA, {1.0f}, {0.15f}, 1.0f);
    Reading r = read(w, rig);
    HostState s;
    w.get_state(s);
    float indent = rig.R - (s.pz[1] - 0.005f);
    std::printf("  uncoupled: W %.4f N, Q %.4f N, pad indents %.4f mm (rigid contact)\n", r.force[2], r.force[0],
                indent * 1e3f);
    CHECK(std::fabs(r.force[2] - 1.0f) < 0.05f);
    CHECK(std::fabs(r.force[0] + 0.15f) < 0.02f);
    CHECK(indent < 1e-5f);
}

}  // namespace

int main() {
    struct Test { const char* name; void (*fn)(); };
    const Test tests[] = {
        {"hertz", test_hertz},
        {"partial_slip", test_partial_slip},
        {"full_slip", test_full_slip},
        {"backends_agree", test_backends_agree},
        {"flat_on_plane", test_flat_on_plane},
        {"coupled_indentation_and_mindlin", test_coupled_indentation_and_mindlin},
        {"coupled_steady", test_coupled_steady},
        {"coupled_corner", test_coupled_corner},
        {"uncoupled", test_uncoupled},
    };
    for (const Test& t : tests) {
        int before = g_failures;
        std::printf("%s\n", t.name);
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  exception: %s\n", e.what());
            g_failures++;
        }
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    std::printf("\n%s (%d failed checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
