#pragma once

// Rigid solver: per-body and per-env routines shared by the CPU and CUDA
// backends. Bodies rotate, have a shape (sphere, capsule, box, plane) and
// friction. Each step is split into substeps; each substep runs
//
//   rigid_apply_actuators    per env   torque / force actuators (if any)
//   rigid_integrate          per body  predict position and orientation
//   rigid_solve_positions    per env   detect contacts; enforce joints and
//                                      position drives, push bodies apart,
//                                      apply static friction
//   rigid_update_velocities  per body  velocities from the position change
//   rigid_solve_velocities   per env   joint damping and velocity drives,
//                                      dynamic friction and restitution
//
// and after the last substep rigid_observe_joint (per joint instance)
// records joint positions and velocities.
//
// following Macklin et al. 2020, "Detailed Rigid Body Simulation with
// Extended Position Based Dynamics". Each env's contacts are solved in
// sequence (Gauss-Seidel) by a single thread, in a fixed order, which keeps
// the result deterministic and identical on CPU and GPU. RL envs hold few
// bodies, so the parallelism comes from running many envs at once.
//
// Conventions: shapes are centred on their body's origin. Capsules run
// along the body's local y axis; planes face along local +y.

#include "phys/common.h"

namespace phys {
namespace detail {

// --- small vector / quaternion helpers ---------------------------------------

struct V3 {
    float x, y, z;
};
struct Q4 {
    float w, x, y, z;
};

PHYS_HD V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
PHYS_HD V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
PHYS_HD V3 operator-(V3 a) { return {-a.x, -a.y, -a.z}; }
PHYS_HD V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
PHYS_HD V3 mul(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
PHYS_HD float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
PHYS_HD V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
PHYS_HD float length(V3 a) { return sqrtf(dot(a, a)); }
PHYS_HD float get(V3 a, int k) { return k == 0 ? a.x : (k == 1 ? a.y : a.z); }
PHYS_HD void set(V3& a, int k, float v) {
    if (k == 0) a.x = v; else if (k == 1) a.y = v; else a.z = v;
}

PHYS_HD Q4 qmul(Q4 a, Q4 b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
PHYS_HD Q4 conj(Q4 q) { return {q.w, -q.x, -q.y, -q.z}; }
PHYS_HD Q4 normalized(Q4 q) {
    float s = 1.0f / sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return {q.w * s, q.x * s, q.y * s, q.z * s};
}
PHYS_HD V3 rotate(Q4 q, V3 v) {
    V3 u = {q.x, q.y, q.z};
    V3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}
PHYS_HD V3 inv_rotate(Q4 q, V3 v) { return rotate(conj(q), v); }
// q + 0.5 * (0, w) * q, renormalised: rotate q by the small angle vector w.
PHYS_HD Q4 rotate_by(Q4 q, V3 w) {
    Q4 dq = qmul(Q4{0.0f, w.x, w.y, w.z}, q);
    return normalized(Q4{q.w + 0.5f * dq.w, q.x + 0.5f * dq.x, q.y + 0.5f * dq.y,
                         q.z + 0.5f * dq.z});
}

// --- buffer access -----------------------------------------------------------

PHYS_HD V3 load3(float* const a[3], int g) { return {a[0][g], a[1][g], a[2][g]}; }
PHYS_HD void store3(float* const a[3], int g, V3 v) {
    a[0][g] = v.x; a[1][g] = v.y; a[2][g] = v.z;
}
PHYS_HD Q4 load4(float* const a[4], int g) { return {a[0][g], a[1][g], a[2][g], a[3][g]}; }
PHYS_HD void store4(float* const a[4], int g, Q4 q) {
    a[0][g] = q.w; a[1][g] = q.x; a[2][g] = q.y; a[3][g] = q.z;
}
PHYS_HD V3 model3(const float* a, int i) { return {a[3 * i], a[3 * i + 1], a[3 * i + 2]}; }

// Body index g is global (env * nbody + body), or -1 for the world, which
// is static and sits at the origin with identity orientation.

PHYS_HD bool is_dynamic(const Params& p, const Buffers& b, int g) {
    return g >= 0 && b.inv_mass[g % p.nbody] > 0.0f;
}

// Current position of body g.
PHYS_HD V3 body_pos(const Buffers& b, int g) { return load3(b.pos, g); }

PHYS_HD Q4 body_quat(const Buffers& b, int g) {
    return g >= 0 ? load4(b.quat, g) : Q4{1.0f, 0.0f, 0.0f, 0.0f};
}

PHYS_HD V3 body_vel(const Buffers& b, int g) {
    return g >= 0 ? load3(b.vel, g) : V3{0.0f, 0.0f, 0.0f};
}

PHYS_HD V3 body_angvel(const Buffers& b, int g) {
    return g >= 0 ? load3(b.angvel, g) : V3{0.0f, 0.0f, 0.0f};
}

// World-frame inverse inertia applied to v.
PHYS_HD V3 inv_inertia_times(const Params& p, const Buffers& b, int g, Q4 q, V3 v) {
    V3 ii = model3(b.inv_inertia, g % p.nbody);
    return rotate(q, mul(ii, inv_rotate(q, v)));
}

// Inverse mass that body g presents to a unit correction along dir at
// offset r from its centre.
PHYS_HD float generalized_inv_mass(const Params& p, const Buffers& b, int g, Q4 q, V3 r,
                                   V3 dir) {
    if (!is_dynamic(p, b, g)) return 0.0f;
    V3 rn = cross(r, dir);
    return b.inv_mass[g % p.nbody] + dot(rn, inv_inertia_times(p, b, g, q, rn));
}

// Inverse moment of inertia of body g about unit axis n.
PHYS_HD float angular_inv_mass(const Params& p, const Buffers& b, int g, V3 n) {
    if (!is_dynamic(p, b, g)) return 0.0f;
    return dot(n, inv_inertia_times(p, b, g, load4(b.quat, g), n));
}

// Move body g by dx and rotate it by the small rotation vector dtheta,
// recording both in the substep's motion.
PHYS_HD void move_body(const Buffers& b, int g, V3 dx, V3 dtheta) {
    store3(b.pos, g, load3(b.pos, g) + dx);
    store3(b.disp, g, load3(b.disp, g) + dx);
    store4(b.quat, g, rotate_by(load4(b.quat, g), dtheta));
    store3(b.drot, g, load3(b.drot, g) + dtheta);
}

// Positional impulse P applied at offset r.
PHYS_HD void apply_position_impulse(const Params& p, const Buffers& b, int g, V3 r, V3 P) {
    if (!is_dynamic(p, b, g)) return;
    move_body(b, g, P * b.inv_mass[g % p.nbody],
              inv_inertia_times(p, b, g, load4(b.quat, g), cross(r, P)));
}

// Pure rotational correction: positional "angular impulse" L.
PHYS_HD void apply_rotation(const Params& p, const Buffers& b, int g, V3 L) {
    if (!is_dynamic(p, b, g)) return;
    move_body(b, g, V3{0.0f, 0.0f, 0.0f}, inv_inertia_times(p, b, g, load4(b.quat, g), L));
}

// Velocity impulse P applied at offset r.
PHYS_HD void apply_velocity_impulse(const Params& p, const Buffers& b, int g, V3 r, V3 P) {
    if (!is_dynamic(p, b, g)) return;
    store3(b.vel, g, load3(b.vel, g) + P * b.inv_mass[g % p.nbody]);
    Q4 q = load4(b.quat, g);
    store3(b.angvel, g, load3(b.angvel, g) + inv_inertia_times(p, b, g, q, cross(r, P)));
}

// Angular impulse L.
PHYS_HD void apply_angular_impulse(const Params& p, const Buffers& b, int g, V3 L) {
    if (!is_dynamic(p, b, g)) return;
    store3(b.angvel, g, load3(b.angvel, g) + inv_inertia_times(p, b, g, load4(b.quat, g), L));
}

// atan2 from a fixed polynomial (max error ~1e-5 rad), so the CPU and GPU
// agree bit for bit; the platforms' atan2f implementations differ.
PHYS_HD float phys_atan2(float y, float x) {
    float ax = fabsf(x), ay = fabsf(y);
    float hi = fmaxf(ax, ay), lo = fminf(ax, ay);
    if (hi == 0.0f) return 0.0f;
    float z = lo / hi, z2 = z * z;
    float a = z * (0.99997726f + z2 * (-0.33262347f + z2 * (0.19354346f +
              z2 * (-0.11643287f + z2 * (0.05265332f + z2 * -0.01172120f)))));
    if (ay > ax) a = 1.57079633f - a;
    if (x < 0.0f) a = 3.14159265f - a;
    return y < 0.0f ? -a : a;
}

// --- per-body phases ---------------------------------------------------------

PHYS_HD void rigid_integrate(const Params& p, const Buffers& b, int g) {
    V3 zero = {0.0f, 0.0f, 0.0f};
    store3(b.disp, g, zero);
    store3(b.drot, g, zero);
    if (!b.enabled[g] || !is_dynamic(p, b, g)) return;

    V3 grav = {p.gravity[0], p.gravity[1], p.gravity[2]};
    V3 v = load3(b.vel, g) + grav * p.h;
    store3(b.vel, g, v);
    Q4 q = load4(b.quat, g);

    // Torque-free rotation, including the gyroscopic term, in the body frame.
    V3 ii = model3(b.inv_inertia, g % p.nbody);
    V3 wb = inv_rotate(q, load3(b.angvel, g));
    V3 Iw = {wb.x / ii.x, wb.y / ii.y, wb.z / ii.z};
    wb = wb + mul(ii, cross(Iw, wb)) * p.h;
    V3 w = rotate(q, wb);
    store3(b.angvel, g, w);
    move_body(b, g, v * p.h, w * p.h);
}

PHYS_HD void rigid_update_velocities(const Params& p, const Buffers& b, int g) {
    if (!b.enabled[g] || !is_dynamic(p, b, g)) return;
    float inv_h = 1.0f / p.h;
    store3(b.vel, g, load3(b.disp, g) * inv_h);
    store3(b.angvel, g, load3(b.drot, g) * inv_h);
}

// --- contact generation ------------------------------------------------------

struct ContactSink {
    const Params& p;
    const Buffers& b;
    Contact* out;  // this env's slice of the contact buffer
    int count;
};

// Record a contact: pa on body a and pb on body b are the touching points,
// n points from b towards a, and (pb - pa) . n is the penetration depth.
PHYS_HD void emit_contact(ContactSink& s, int a, int c, V3 pa, V3 pb, V3 n) {
    if (s.count < s.p.max_contacts) {
        const Params& p = s.p;
        const Buffers& b = s.b;
        V3 xa = body_pos(b, a), xb = body_pos(b, c);
        Q4 qa = load4(b.quat, a), qb = load4(b.quat, c);
        V3 ra = inv_rotate(qa, pa - xa);
        V3 rb = inv_rotate(qb, pb - xb);
        V3 va = load3(b.vel, a) + cross(load3(b.angvel, a), pa - xa);
        V3 vb = load3(b.vel, c) + cross(load3(b.angvel, c), pb - xb);
        int ia = a % p.nbody, ib = c % p.nbody;

        Contact& k = s.out[s.count];
        k.a = a;
        k.b = c;
        k.n[0] = n.x; k.n[1] = n.y; k.n[2] = n.z;
        k.ra[0] = ra.x; k.ra[1] = ra.y; k.ra[2] = ra.z;
        k.rb[0] = rb.x; k.rb[1] = rb.y; k.rb[2] = rb.z;
        k.lambda_n = 0.0f;
        k.static_friction[0] = k.static_friction[1] = k.static_friction[2] = 0.0f;
        k.friction_impulse[0] = k.friction_impulse[1] = k.friction_impulse[2] = 0.0f;
        k.normal_impulse = 0.0f;
        k.pad = -1;
        k.compliance = 0.0f;
        k.vn_pre = dot(va - vb, n);
        k.mu = 0.5f * (b.friction[ia] + b.friction[ib]);
        k.e = 0.5f * (b.restitution[ia] + b.restitution[ib]);
    }
    s.count++;
}

PHYS_HD void spheres(ContactSink& s, int a, V3 ca, float ra, int c, V3 cb, float rb) {
    V3 d = ca - cb;
    float dist_sq = dot(d, d);
    float rsum = ra + rb;
    if (dist_sq >= rsum * rsum) return;
    float dist = sqrtf(dist_sq);
    V3 n = dist > 1e-6f ? d * (1.0f / dist) : V3{0.0f, 1.0f, 0.0f};
    emit_contact(s, a, c, ca - n * ra, cb + n * rb, n);
}

PHYS_HD void sphere_plane(ContactSink& s, int a, V3 ca, float r, int c) {
    V3 n = rotate(load4(s.b.quat, c), V3{0.0f, 1.0f, 0.0f});
    float dist = dot(ca - load3(s.b.pos, c), n);
    if (dist >= r) return;
    emit_contact(s, a, c, ca - n * r, ca - n * dist, n);
}

// Sphere at ca (part of body a) against box body c.
PHYS_HD void sphere_box(ContactSink& s, int a, V3 ca, float r, int c) {
    V3 xb = load3(s.b.pos, c);
    Q4 qb = load4(s.b.quat, c);
    V3 half = model3(s.b.size, c % s.p.nbody);
    V3 local = inv_rotate(qb, ca - xb);
    V3 closest = local;
    bool inside = true;
    for (int k = 0; k < 3; k++) {
        float v = get(local, k), hk = get(half, k);
        if (v < -hk) { set(closest, k, -hk); inside = false; }
        if (v > hk) { set(closest, k, hk); inside = false; }
    }
    V3 n_local;
    if (!inside) {
        V3 d = local - closest;
        float dist_sq = dot(d, d);
        if (dist_sq >= r * r) return;
        n_local = d * (1.0f / sqrtf(dist_sq));
    } else {
        // Centre inside the box: push out through the nearest face.
        int axis = 0;
        float best = half.x - fabsf(local.x);
        for (int k = 1; k < 3; k++) {
            float depth = get(half, k) - fabsf(get(local, k));
            if (depth < best) { best = depth; axis = k; }
        }
        float sign = get(local, axis) >= 0.0f ? 1.0f : -1.0f;
        n_local = V3{0.0f, 0.0f, 0.0f};
        set(n_local, axis, sign);
        set(closest, axis, sign * get(half, axis));
    }
    V3 n = rotate(qb, n_local);
    emit_contact(s, a, c, ca - n * r, xb + rotate(qb, closest), n);
}

PHYS_HD void capsule_ends(const Params& p, const Buffers& b, int g, V3& e0, V3& e1) {
    V3 x = body_pos(b, g);
    V3 axis = rotate(load4(b.quat, g), V3{0.0f, b.size[3 * (g % p.nbody) + 1], 0.0f});
    e0 = x - axis;
    e1 = x + axis;
}

PHYS_HD V3 closest_on_segment(V3 a, V3 b, V3 pt) {
    V3 ab = b - a;
    float len_sq = dot(ab, ab);
    float t = len_sq > 0.0f ? dot(pt - a, ab) / len_sq : 0.0f;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return a + ab * t;
}

// Closest points between segments p0-p1 and q0-q1 (Ericson, RTCD 5.1.9).
PHYS_HD void closest_between_segments(V3 p0, V3 p1, V3 q0, V3 q1, V3& cp, V3& cq) {
    V3 d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    float s = 0.0f, t = 0.0f;
    if (a <= 1e-12f && e <= 1e-12f) {
        // both degenerate
    } else if (a <= 1e-12f) {
        t = fminf(fmaxf(f / e, 0.0f), 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= 1e-12f) {
            s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
        } else {
            float bb = dot(d1, d2);
            float denom = a * e - bb * bb;
            s = denom > 1e-12f ? fminf(fmaxf((bb * f - c * e) / denom, 0.0f), 1.0f) : 0.0f;
            t = (bb * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = fminf(fmaxf(-c / a, 0.0f), 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = fminf(fmaxf((bb - c) / a, 0.0f), 1.0f);
            }
        }
    }
    cp = p0 + d1 * s;
    cq = q0 + d2 * t;
}

PHYS_HD V3 box_vertex(V3 x, Q4 q, V3 half, int v) {
    V3 corner = {(v & 1) ? half.x : -half.x, (v & 2) ? half.y : -half.y,
                 (v & 4) ? half.z : -half.z};
    return x + rotate(q, corner);
}

PHYS_HD void box_plane(ContactSink& s, int a, int c) {
    V3 xa = load3(s.b.pos, a);
    Q4 qa = load4(s.b.quat, a);
    V3 half = model3(s.b.size, a % s.p.nbody);
    V3 n = rotate(load4(s.b.quat, c), V3{0.0f, 1.0f, 0.0f});
    V3 xp = load3(s.b.pos, c);
    for (int v = 0; v < 8; v++) {
        V3 pt = box_vertex(xa, qa, half, v);
        float dist = dot(pt - xp, n);
        if (dist < 0.0f) emit_contact(s, a, c, pt, pt - n * dist, n);
    }
}

struct Box {
    V3 c;       // centre
    V3 ax[3];   // world-frame axes
    float h[3]; // half extents
};

PHYS_HD Box load_box(const Params& p, const Buffers& b, int g) {
    Box box;
    box.c = body_pos(b, g);
    Q4 q = load4(b.quat, g);
    box.ax[0] = rotate(q, V3{1.0f, 0.0f, 0.0f});
    box.ax[1] = rotate(q, V3{0.0f, 1.0f, 0.0f});
    box.ax[2] = rotate(q, V3{0.0f, 0.0f, 1.0f});
    V3 half = model3(b.size, g % p.nbody);
    box.h[0] = half.x; box.h[1] = half.y; box.h[2] = half.z;
    return box;
}

// Half-width of a box's projection onto unit axis L.
PHYS_HD float box_extent(const Box& box, V3 L) {
    return box.h[0] * fabsf(dot(box.ax[0], L)) + box.h[1] * fabsf(dot(box.ax[1], L)) +
           box.h[2] * fabsf(dot(box.ax[2], L));
}

// Box-box contact manifold. A separating-axis test over the 15 candidate
// axes finds the axis of least penetration. For a face axis, the most
// anti-parallel face of the other ("incident") box is clipped against the
// side planes of the reference face, giving up to 8 contacts; for an edge
// axis, the closest points between the two edges give one.
PHYS_HD void box_box(ContactSink& s, int a, int c) {
    Box A = load_box(s.p, s.b, a), B = load_box(s.p, s.b, c);
    V3 T = B.c - A.c;

    // Least-penetration axis. Ties are broken in favour of A's faces, then B's,
    // then edges, with small tolerances, so that resting contact keeps the
    // same reference face from substep to substep instead of flipping on
    // rounding noise.
    float best_sep = -1e30f;
    int best = -1;  // 0-2: face of A, 3-5: face of B, 6-14: edge pair
    V3 best_axis = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 15; i++) {
        V3 L;
        if (i < 3) {
            L = A.ax[i];
        } else if (i < 6) {
            L = B.ax[i - 3];
        } else {
            L = cross(A.ax[(i - 6) / 3], B.ax[(i - 6) % 3]);
            float len = length(L);
            if (len < 1e-5f) continue;  // parallel edges: covered by the face axes
            L = L * (1.0f / len);
        }
        float sep = fabsf(dot(T, L)) - box_extent(A, L) - box_extent(B, L);
        if (sep > 0.0f) return;  // separating axis found
        float bias = i < 3 ? 0.0f : (i < 6 ? 1e-4f : 1e-3f + 0.05f * fabsf(best_sep));
        if (sep > best_sep + bias) {
            best_sep = sep;
            best = i;
            best_axis = dot(T, L) < 0.0f ? -L : L;  // oriented from A towards B
        }
    }

    if (best >= 6) {
        // Edge-edge: closest points between the supporting edges.
        int i = (best - 6) / 3, j = (best - 6) % 3;
        V3 n = best_axis;
        V3 ea = A.c, eb = B.c;
        for (int k = 0; k < 3; k++) {
            if (k != i) ea = ea + A.ax[k] * (dot(A.ax[k], n) > 0.0f ? A.h[k] : -A.h[k]);
            if (k != j) eb = eb + B.ax[k] * (dot(B.ax[k], n) > 0.0f ? -B.h[k] : B.h[k]);
        }
        V3 pa, pb;
        closest_between_segments(ea - A.ax[i] * A.h[i], ea + A.ax[i] * A.h[i],
                                 eb - B.ax[j] * B.h[j], eb + B.ax[j] * B.h[j], pa, pb);
        emit_contact(s, a, c, pa, pb, -n);
        return;
    }

    // Face contact: R is the reference box, I the incident one.
    bool ref_is_a = best < 3;
    const Box& R = ref_is_a ? A : B;
    const Box& I = ref_is_a ? B : A;
    int rk = ref_is_a ? best : best - 3;
    V3 nr = ref_is_a ? best_axis : -best_axis;  // reference normal, towards I
    float rsign = dot(R.ax[rk], nr) > 0.0f ? 1.0f : -1.0f;

    // Incident face: the face of I most opposed to nr.
    int ik = 0;
    float best_dot = 0.0f;
    for (int k = 0; k < 3; k++) {
        float d = fabsf(dot(I.ax[k], nr));
        if (d > best_dot) { best_dot = d; ik = k; }
    }
    float isign = dot(I.ax[ik], nr) > 0.0f ? -1.0f : 1.0f;
    int iu = (ik + 1) % 3, iv = (ik + 2) % 3;
    V3 face_c = I.c + I.ax[ik] * (isign * I.h[ik]);
    V3 du = I.ax[iu] * I.h[iu], dv = I.ax[iv] * I.h[iv];

    V3 poly[8], next[8];
    int count = 4;
    poly[0] = face_c - du - dv;
    poly[1] = face_c + du - dv;
    poly[2] = face_c + du + dv;
    poly[3] = face_c - du + dv;

    // Clip against the four side planes of the reference face, widened by a
    // small tolerance: equal-sized boxes stacked flush have incident edges
    // lying exactly on those planes, and rounding must not discard them.
    for (int side = 0; side < 4 && count > 0; side++) {
        int k = side < 2 ? (rk + 1) % 3 : (rk + 2) % 3;
        float sgn = (side & 1) ? -1.0f : 1.0f;
        V3 pn = R.ax[k] * sgn;
        float offset = dot(R.c, pn) + R.h[k] * 1.001f + 1e-5f;
        int out = 0;
        for (int v = 0; v < count; v++) {
            V3 p0 = poly[v], p1 = poly[(v + 1) % count];
            float d0 = dot(p0, pn) - offset, d1 = dot(p1, pn) - offset;
            if (d0 <= 0.0f && out < 8) next[out++] = p0;
            if ((d0 <= 0.0f) != (d1 <= 0.0f) && out < 8)
                next[out++] = p0 + (p1 - p0) * (d0 / (d0 - d1));
        }
        count = out;
        for (int v = 0; v < count; v++) poly[v] = next[v];
    }

    // Keep the clipped points below the reference face.
    V3 face_n = R.ax[rk] * rsign;
    float face_offset = dot(R.c, face_n) + R.h[rk];
    for (int v = 0; v < count; v++) {
        float depth = face_offset - dot(poly[v], face_n);
        if (depth <= 0.0f) continue;
        V3 on_ref = poly[v] + face_n * depth;
        if (ref_is_a)
            emit_contact(s, a, c, on_ref, poly[v], -face_n);
        else
            emit_contact(s, a, c, poly[v], on_ref, face_n);
    }
}

PHYS_HD void collide_pair(ContactSink& s, int i, int j) {
    const Params& p = s.p;
    const Buffers& b = s.b;
    int si = b.shape[i % p.nbody], sj = b.shape[j % p.nbody];
    if (si == kNoShape || sj == kNoShape) return;
    if (!is_dynamic(p, b, i) && !is_dynamic(p, b, j)) return;
    if (!b.may_collide[(i % p.nbody) * p.nbody + j % p.nbody]) return;
    if (si != kPlane && sj != kPlane) {
        V3 d = body_pos(b, i) - body_pos(b, j);
        float reach = b.radius[i % p.nbody] + b.radius[j % p.nbody];
        if (dot(d, d) > reach * reach) return;
    }
    // Order the pair so that a's shape type <= c's (planes last).
    int a = i, c = j;
    if (si > sj) {
        a = j; c = i;
        int t = si; si = sj; sj = t;
    }
    float ra = b.size[3 * (a % p.nbody)];
    float rc = b.size[3 * (c % p.nbody)];
    V3 a0, a1, c0, c1;

    if (si == kSphere) {
        V3 ca = body_pos(b, a);
        if (sj == kSphere) {
            spheres(s, a, ca, ra, c, body_pos(b, c), rc);
        } else if (sj == kCapsule) {
            capsule_ends(p, b, c, c0, c1);
            spheres(s, a, ca, ra, c, closest_on_segment(c0, c1, ca), rc);
        } else if (sj == kBox) {
            sphere_box(s, a, ca, ra, c);
        } else {
            sphere_plane(s, a, ca, ra, c);
        }
    } else if (si == kCapsule) {
        capsule_ends(p, b, a, a0, a1);
        if (sj == kCapsule) {
            capsule_ends(p, b, c, c0, c1);
            V3 pa, pc;
            closest_between_segments(a0, a1, c0, c1, pa, pc);
            spheres(s, a, pa, ra, c, pc, rc);
        } else if (sj == kBox) {
            // Approximation: the capsule's two end spheres and its midpoint.
            sphere_box(s, a, a0, ra, c);
            sphere_box(s, a, a1, ra, c);
            sphere_box(s, a, (a0 + a1) * 0.5f, ra, c);
        } else {
            sphere_plane(s, a, a0, ra, c);
            sphere_plane(s, a, a1, ra, c);
        }
    } else if (si == kBox) {
        if (sj == kBox) {
            box_box(s, a, c);
        } else {
            box_plane(s, a, c);
        }
    }
    // plane-plane: never in contact
}

// --- joints ------------------------------------------------------------------

// A joint frame in world coordinates: its origin, orientation, and the
// origin's offset from the body's centre (unused for the world).
struct JointFrame {
    V3 point;
    Q4 rot;
    V3 r;
};

PHYS_HD JointFrame joint_frame(const Buffers& b, int g, const float anchor[3],
                               const float frame[4]) {
    V3 a = {anchor[0], anchor[1], anchor[2]};
    Q4 f = {frame[0], frame[1], frame[2], frame[3]};
    if (g < 0) return {a, f, V3{0.0f, 0.0f, 0.0f}};
    Q4 q = load4(b.quat, g);
    V3 r = rotate(q, a);
    return {body_pos(b, g) + r, qmul(q, f), r};
}

PHYS_HD V3 frame_axis(Q4 rot, int k) {
    return rotate(rot, V3{k == 0 ? 1.0f : 0.0f, k == 1 ? 1.0f : 0.0f, k == 2 ? 1.0f : 0.0f});
}

// Signed hinge angle: from the parent frame's y axis to the child's, about
// the parent frame's x axis.
PHYS_HD float hinge_angle(Q4 parent_rot, Q4 child_rot) {
    V3 ex = frame_axis(parent_rot, 0);
    V3 yp = frame_axis(parent_rot, 1), yc = frame_axis(child_rot, 1);
    return phys_atan2(dot(cross(yp, yc), ex), dot(yp, yc));
}

// Move body a's point at offset ra by `corr` relative to body b's point at
// rb (a rigid, zero-compliance positional constraint).
PHYS_HD void correct_point(const Params& p, const Buffers& b, int ga, V3 ra, int gb, V3 rb,
                           V3 corr) {
    float mag = length(corr);
    if (mag <= 1e-20f) return;
    V3 dir = corr * (1.0f / mag);
    float w = generalized_inv_mass(p, b, ga, body_quat(b, ga), ra, dir) +
              generalized_inv_mass(p, b, gb, body_quat(b, gb), rb, dir);
    if (w <= 0.0f) return;
    V3 P = dir * (mag / w);
    apply_position_impulse(p, b, ga, ra, P);
    apply_position_impulse(p, b, gb, rb, -P);
}

// Rotate body a by the small rotation vector `corr` relative to body b.
PHYS_HD void correct_rotation(const Params& p, const Buffers& b, int ga, int gb, V3 corr) {
    float mag = length(corr);
    if (mag <= 1e-20f) return;
    V3 n = corr * (1.0f / mag);
    float w = angular_inv_mass(p, b, ga, n) + angular_inv_mass(p, b, gb, n);
    if (w <= 0.0f) return;
    V3 L = n * (mag / w);
    apply_rotation(p, b, ga, L);
    apply_rotation(p, b, gb, -L);
}

// Compliant (XPBD) drive step for constraint value C along unit direction n:
// rotational when `angular`, else positional at offsets ra / rb. lambda is
// the correction accumulated this substep; |lambda| is capped at max_lambda
// (when positive), which bounds the drive's force.
PHYS_HD void drive_step(const Params& p, const Buffers& b, int ga, V3 ra, int gb, V3 rb, V3 n,
                        bool angular, float C, float alpha, float max_lambda, float& lambda) {
    float w = angular ? angular_inv_mass(p, b, ga, n) + angular_inv_mass(p, b, gb, n)
                      : generalized_inv_mass(p, b, ga, body_quat(b, ga), ra, n) +
                            generalized_inv_mass(p, b, gb, body_quat(b, gb), rb, n);
    if (w + alpha <= 0.0f) return;
    float total = lambda + (-C - alpha * lambda) / (w + alpha);
    if (max_lambda > 0.0f) total = fminf(fmaxf(total, -max_lambda), max_lambda);
    float dl = total - lambda;
    lambda = total;
    if (angular) {
        apply_rotation(p, b, ga, n * dl);
        apply_rotation(p, b, gb, n * -dl);
    } else {
        apply_position_impulse(p, b, ga, ra, n * dl);
        apply_position_impulse(p, b, gb, rb, n * -dl);
    }
}

PHYS_HD void joint_bodies(const Params& p, const JointModel& jm, int env, int& gc, int& gp) {
    gc = env * p.nbody + jm.child;
    gp = jm.parent < 0 ? -1 : env * p.nbody + jm.parent;
}

PHYS_HD bool joint_active(const Buffers& b, int gc, int gp) {
    return b.enabled[gc] && (gp < 0 || b.enabled[gp]);
}

// One position-level pass over joint j of env: orientation constraint,
// anchor constraint, limits, then the position drive.
PHYS_HD void solve_joint_position(const Params& p, const Buffers& b, int env, int j) {
    const JointModel& jm = b.joints[j];
    int gc, gp;
    joint_bodies(p, jm, env, gc, gp);
    if (!joint_active(b, gc, gp)) return;

    // Orientation: hinges keep their axes aligned; sliders and fixed joints
    // keep the frames' orientations equal; ball joints leave it free.
    JointFrame fp = joint_frame(b, gp, jm.parent_anchor, jm.parent_frame);
    JointFrame fc = joint_frame(b, gc, jm.child_anchor, jm.child_frame);
    if (jm.type == kHinge) {
        correct_rotation(p, b, gc, gp, cross(frame_axis(fc.rot, 0), frame_axis(fp.rot, 0)));
    } else if (jm.type == kSlider || jm.type == kFixed) {
        Q4 dq = qmul(fp.rot, conj(fc.rot));
        V3 v = V3{dq.x, dq.y, dq.z} * 2.0f;
        correct_rotation(p, b, gc, gp, dq.w >= 0.0f ? v : -v);
    }

    // Anchors coincide, except along a slider's axis (within its limits).
    fp = joint_frame(b, gp, jm.parent_anchor, jm.parent_frame);
    fc = joint_frame(b, gc, jm.child_anchor, jm.child_frame);
    V3 delta = fp.point - fc.point;
    if (jm.type == kSlider) {
        V3 ex = frame_axis(fp.rot, 0);
        // The lateral error, with the axial part removed exactly: in the
        // joint frame the axis is x, so zero x there. Projecting with the
        // float axis instead (delta - ex (ex . delta)) leaves an axial part
        // of ~1e-7 times the slider's travel - the same tiny push every
        // substep, which feeds momentum along the axis and biases every
        // force balance on it.
        V3 local = inv_rotate(fp.rot, delta);
        float d = -local.x;  // child offset along the axis
        local.x = 0.0f;
        V3 corr = rotate(fp.rot, local);
        if (jm.limited) {
            if (d < jm.lower) corr = corr + ex * (jm.lower - d);
            if (d > jm.upper) corr = corr + ex * (jm.upper - d);
        }
        correct_point(p, b, gc, fc.r, gp, fp.r, corr);
    } else {
        correct_point(p, b, gc, fc.r, gp, fp.r, delta);
    }

    if (jm.type == kHinge && jm.limited) {
        fp = joint_frame(b, gp, jm.parent_anchor, jm.parent_frame);
        fc = joint_frame(b, gc, jm.child_anchor, jm.child_frame);
        float angle = hinge_angle(fp.rot, fc.rot);
        V3 ex = frame_axis(fp.rot, 0);
        if (angle < jm.lower) correct_rotation(p, b, gc, gp, ex * (jm.lower - angle));
        if (angle > jm.upper) correct_rotation(p, b, gc, gp, ex * (jm.upper - angle));
    }

    // Position drive: a compliant constraint pulling the joint to its
    // target, with compliance 1 / kp. Unconditionally stable, unlike an
    // explicit PD torque, even for light links and stiff gains.
    if (jm.actuator == kPositionDrive && jm.kp > 0.0f &&
        (jm.type == kHinge || jm.type == kSlider)) {
        float target = b.ctrl[env * p.njoint + j];
        if (jm.limited) target = fminf(fmaxf(target, jm.lower), jm.upper);
        float alpha = 1.0f / (jm.kp * p.h * p.h);
        float max_lambda = jm.max_force * p.h * p.h;
        float& lambda = b.joint_lambda[env * p.njoint + j];
        fp = joint_frame(b, gp, jm.parent_anchor, jm.parent_frame);
        fc = joint_frame(b, gc, jm.child_anchor, jm.child_frame);
        V3 ex = frame_axis(fp.rot, 0);
        if (jm.type == kHinge) {
            float C = hinge_angle(fp.rot, fc.rot) - target;
            drive_step(p, b, gc, fc.r, gp, fp.r, ex, true, C, alpha, max_lambda, lambda);
        } else {
            float C = dot(fc.point - fp.point, ex) - target;
            drive_step(p, b, gc, fc.r, gp, fp.r, ex, false, C, alpha, max_lambda, lambda);
        }
    }
}

// Joint velocity along its free axis: relative angular velocity about a
// hinge axis, or relative linear velocity of the anchors along a slider.
PHYS_HD float joint_velocity(const Buffers& b, const JointModel& jm, int gc, int gp,
                             const JointFrame& fc, const JointFrame& fp) {
    V3 ex = frame_axis(fp.rot, 0);
    if (jm.type == kHinge) return dot(body_angvel(b, gc) - body_angvel(b, gp), ex);
    V3 vc = body_vel(b, gc) + cross(body_angvel(b, gc), fc.r);
    V3 vp = body_vel(b, gp) + cross(body_angvel(b, gp), fp.r);
    return dot(vc - vp, ex);
}

// Velocity-level joint work: passive damping, the position drive's damping
// (kd) and velocity drives.
PHYS_HD void solve_joint_velocity(const Params& p, const Buffers& b, int env, int j) {
    const JointModel& jm = b.joints[j];
    if (jm.type != kHinge && jm.type != kSlider) return;
    int gc, gp;
    joint_bodies(p, jm, env, gc, gp);
    if (!joint_active(b, gc, gp)) return;
    float damping = jm.damping + (jm.actuator == kPositionDrive ? jm.kd : 0.0f);
    bool velocity_drive = jm.actuator == kVelocityDrive;
    if (damping <= 0.0f && !velocity_drive) return;

    JointFrame fp = joint_frame(b, gp, jm.parent_anchor, jm.parent_frame);
    JointFrame fc = joint_frame(b, gc, jm.child_anchor, jm.child_frame);
    V3 ex = frame_axis(fp.rot, 0);
    bool angular = jm.type == kHinge;
    float w = angular ? angular_inv_mass(p, b, gc, ex) + angular_inv_mass(p, b, gp, ex)
                      : generalized_inv_mass(p, b, gc, body_quat(b, gc), fc.r, ex) +
                            generalized_inv_mass(p, b, gp, body_quat(b, gp), fp.r, ex);
    if (w <= 0.0f) return;

    float impulse = 0.0f;
    float v = joint_velocity(b, jm, gc, gp, fc, fp);
    if (damping > 0.0f) {
        // Implicit-style damping: never more than what stops the joint.
        impulse = -damping * v * p.h;
        float stop = fabsf(v) / w;
        impulse = fminf(fmaxf(impulse, -stop), stop);
        v += impulse * w;
    }
    if (velocity_drive) {
        float drive = (b.ctrl[env * p.njoint + j] - v) / w;
        if (jm.max_force > 0.0f) {
            float cap = jm.max_force * p.h;
            drive = fminf(fmaxf(drive, -cap), cap);
        }
        impulse += drive;
    }
    if (angular) {
        apply_angular_impulse(p, b, gc, ex * impulse);
        apply_angular_impulse(p, b, gp, ex * -impulse);
    } else {
        apply_velocity_impulse(p, b, gc, fc.r, ex * impulse);
        apply_velocity_impulse(p, b, gp, fp.r, ex * -impulse);
    }
}

// Torque / force actuators, applied as an impulse before integration.
PHYS_HD void rigid_apply_actuators(const Params& p, const Buffers& b, int env) {
    for (int j = 0; j < p.njoint; j++) {
        const JointModel& jm = b.joints[j];
        if (jm.actuator != kTorque || (jm.type != kHinge && jm.type != kSlider)) continue;
        int gc, gp;
        joint_bodies(p, jm, env, gc, gp);
        if (!joint_active(b, gc, gp)) continue;
        float u = b.ctrl[env * p.njoint + j];
        if (jm.max_force > 0.0f) u = fminf(fmaxf(u, -jm.max_force), jm.max_force);
        JointFrame fp = joint_frame(b, gp, jm.parent_anchor, jm.parent_frame);
        JointFrame fc = joint_frame(b, gc, jm.child_anchor, jm.child_frame);
        V3 J = frame_axis(fp.rot, 0) * (u * p.h);
        if (jm.type == kHinge) {
            apply_angular_impulse(p, b, gc, J);
            apply_angular_impulse(p, b, gp, -J);
        } else {
            apply_velocity_impulse(p, b, gc, fc.r, J);
            apply_velocity_impulse(p, b, gp, fp.r, -J);
        }
    }
}

// Joint position and velocity of joint instance idx = env * njoint + j.
PHYS_HD void rigid_observe_joint(const Params& p, const Buffers& b, int idx) {
    int env = idx / p.njoint, j = idx % p.njoint;
    const JointModel& jm = b.joints[j];
    float q = 0.0f, qd = 0.0f;
    if (jm.type == kHinge || jm.type == kSlider) {
        int gc, gp;
        joint_bodies(p, jm, env, gc, gp);
        JointFrame fp = joint_frame(b, gp, jm.parent_anchor, jm.parent_frame);
        JointFrame fc = joint_frame(b, gc, jm.child_anchor, jm.child_frame);
        q = jm.type == kHinge ? hinge_angle(fp.rot, fc.rot)
                              : dot(fc.point - fp.point, frame_axis(fp.rot, 0));
        qd = joint_velocity(b, jm, gc, gp, fc, fp);
    }
    b.joint_q[idx] = q;
    b.joint_qd[idx] = qd;
}

// --- per-env phases ----------------------------------------------------------

PHYS_HD void contact_points(const Buffers& b, const Contact& k, V3& pa, V3& pb, V3& ra,
                            V3& rb) {
    ra = rotate(load4(b.quat, k.a), V3{k.ra[0], k.ra[1], k.ra[2]});
    rb = rotate(load4(b.quat, k.b), V3{k.rb[0], k.rb[1], k.rb[2]});
    pa = body_pos(b, k.a) + ra;
    pb = body_pos(b, k.b) + rb;
}

// Velocity impulse of size `magnitude` along dir (a gets +dir, b -dir).
PHYS_HD void apply_contact_impulse(const Params& p, const Buffers& b, const Contact& k, V3 ra,
                                   V3 rb, V3 P) {
    apply_velocity_impulse(p, b, k.a, ra, P);
    apply_velocity_impulse(p, b, k.b, rb, -P);
}

PHYS_HD V3 relative_velocity(const Buffers& b, const Contact& k, V3 ra, V3 rb) {
    V3 va = load3(b.vel, k.a) + cross(load3(b.angvel, k.a), ra);
    V3 vb = load3(b.vel, k.b) + cross(load3(b.angvel, k.b), rb);
    return va - vb;
}

// --- tactile pads in the rigid solve -----------------------------------------
//
// Contacts on the face of a tactile pad (phys::TactileSensorDesc) whose
// sensor is `coupled` take the gel's compliance:
//   normal      a compliant constraint whose stiffness is the gel's secant
//               stiffness W / delta from the last tactile solve, shared by
//               the pad's contacts, so the bodies indent the gel as far as
//               the gel model says;
//   tangential  instead of rigid static / dynamic friction, a persistent
//               shear deflection u of the gel with Mindlin's force law
//               Q = mu W (1 - (1 - |u| / u*)^(3/2)), u* = 3 mu W / (2 k_t),
//               critically damped while it sticks; beyond u* the pad slides.
// The tactile solve (tactile_sensor.h) uses the same theory for the stick
// zone, and refreshes the stiffnesses from the contact it finds.

struct PadFrame {
    V3 origin, tx, ty, nz;
};

PHYS_HD PadFrame pad_frame(const Buffers& b, const TactileModel& s, int g) {
    Q4 qb = load4(b.quat, g);
    Q4 q = qmul(qb, Q4{s.frame[0], s.frame[1], s.frame[2], s.frame[3]});
    PadFrame f;
    f.origin = body_pos(b, g) + rotate(qb, V3{s.origin[0], s.origin[1], s.origin[2]});
    f.tx = rotate(q, V3{1.0f, 0.0f, 0.0f});
    f.ty = rotate(q, V3{0.0f, 1.0f, 0.0f});
    f.nz = rotate(q, V3{0.0f, 0.0f, 1.0f});
    return f;
}

// The sensor whose pad contact c presses on, or -1: the contact involves the
// pad's body and pushes it against the pad normal, at most one pad size
// beyond the pad's edges (the gel is taken to cover the face around the pad,
// so an indented contact that slides off the sensing area does not turn
// rigid at once). The nearest pad wins.
PHYS_HD int pad_of_contact(const Params& p, const Buffers& b, const Contact& c) {
    int env = c.a / p.nbody;
    int best = -1;
    float best_out = 0.0f;
    for (int s = 0; s < p.nsensor; s++) {
        const TactileModel& m = b.sensors[s];
        int g = env * p.nbody + m.body;
        if (c.a != g && c.b != g) continue;
        bool on_a = c.a == g;
        V3 n = {c.n[0], c.n[1], c.n[2]};
        PadFrame f = pad_frame(b, m, g);
        if (dot(on_a ? n : -n, f.nz) > -0.5f) continue;  // not on the pad's face
        V3 pa, pb, ra, rb;
        contact_points(b, c, pa, pb, ra, rb);
        V3 rel = (on_a ? pa : pb) - f.origin;
        float ox = fmaxf(fabsf(dot(rel, f.tx)) - m.half_w, 0.0f), oy = fmaxf(fabsf(dot(rel, f.ty)) - m.half_h, 0.0f);
        float out = sqrtf(ox * ox + oy * oy);  // distance outside the pad
        if (out > 2.0f * fmaxf(m.half_w, m.half_h)) continue;
        if (best < 0 || out < best_out) {
            best = s;
            best_out = out;
        }
    }
    return best;
}

PHYS_HD float* pad_state(const Params& p, const Buffers& b, int env, int s) {
    return b.tactile_coupling + ((long long)env * p.nsensor + s) * kPadState;
}

// Tag this substep's contacts with their pads and give the contacts on
// coupled pads their share of the gel's normal compliance.
PHYS_HD void tag_pad_contacts(const Params& p, const Buffers& b, int env, Contact* contacts, int n) {
    for (int k = 0; k < n; k++) contacts[k].pad = pad_of_contact(p, b, contacts[k]);
    for (int k = 0; k < n; k++) {
        int s = contacts[k].pad;
        if (s < 0 || !b.sensors[s].coupled) continue;
        int share = 0;
        for (int j = 0; j < n; j++) share += contacts[j].pad == s ? 1 : 0;
        float stiffness = pad_state(p, b, env, s)[kPadNormalStiffness];
        if (!(stiffness > 0.0f)) stiffness = b.sensors[s].default_stiffness;
        contacts[k].compliance = (float)share / stiffness;
    }
}

// The gel's shear on every coupled pad of env `env`, applied as velocity
// impulses after the position solve of a substep.
PHYS_HD void solve_pad_shear(const Params& p, const Buffers& b, int env) {
    const Contact* contacts = b.contacts + (long long)env * p.max_contacts;
    int n = b.ncontact[env] < p.max_contacts ? b.ncontact[env] : p.max_contacts;
    float inv_h2 = 1.0f / (p.h * p.h);
    for (int s = 0; s < p.nsensor; s++) {
        const TactileModel& m = b.sensors[s];
        if (!m.coupled) continue;
        float* st = pad_state(p, b, env, s);
        int g = env * p.nbody + m.body;
        // Load, friction coefficient and centre of the pad's contacts, and
        // the body pressing hardest.
        float lambda = 0.0f, mu = 0.0f, best = 0.0f;
        V3 centre = {0.0f, 0.0f, 0.0f};
        int other = -1;
        for (int k = 0; k < n; k++) {
            const Contact& c = contacts[k];
            if (c.pad != s || c.lambda_n <= 0.0f) continue;
            V3 pa, pb, ra, rb;
            contact_points(b, c, pa, pb, ra, rb);
            centre = centre + (c.a == g ? pa : pb) * c.lambda_n;
            lambda += c.lambda_n;
            mu += c.mu * c.lambda_n;
            if (c.lambda_n > best) {
                best = c.lambda_n;
                other = c.a == g ? c.b : c.a;
            }
        }
        st[kPadForce] = st[kPadForce + 1] = st[kPadForce + 2] = 0.0f;
        // Critical damping of the gel's normal motion at each contact, in
        // place of restitution; it may slow separation but never makes the
        // contact pull.
        for (int k = 0; k < n; k++) {
            Contact& c = b.contacts[(long long)env * p.max_contacts + k];
            if (c.pad != s || c.lambda_n <= 0.0f || c.compliance <= 0.0f) continue;
            V3 nn = {c.n[0], c.n[1], c.n[2]};
            V3 pa, pb, ra, rb;
            contact_points(b, c, pa, pb, ra, rb);
            float vn = dot(relative_velocity(b, c, ra, rb), nn);
            float w = generalized_inv_mass(p, b, c.a, load4(b.quat, c.a), ra, nn) +
                      generalized_inv_mass(p, b, c.b, load4(b.quat, c.b), rb, nn);
            if (w <= 0.0f) continue;
            float damping = 2.0f * sqrtf(1.0f / (c.compliance * w));
            float j = -damping * vn * p.h;
            j = fmaxf(fminf(j, fabsf(vn) / w), -fabsf(vn) / w);   // never reverse the motion
            j = fmaxf(j, -c.lambda_n / p.h);                       // never pull
            apply_contact_impulse(p, b, c, ra, rb, nn * j);
            c.normal_impulse += j;
        }
        if (lambda <= 0.0f || other < 0) {
            st[kPadShearX] = st[kPadShearY] = 0.0f;
            continue;
        }
        centre = centre * (1.0f / lambda);
        mu /= lambda;
        float W = lambda * inv_h2;
        PadFrame f = pad_frame(b, m, g);
        V3 rp = centre - body_pos(b, g), ro = centre - body_pos(b, other);
        // The gel shears by the object's motion relative to the pad over the
        // substep (velocities are the substep's displacement / h).
        V3 dp = (body_vel(b, g) + cross(body_angvel(b, g), rp)) * p.h;
        V3 dobj = (body_vel(b, other) + cross(body_angvel(b, other), ro)) * p.h;
        V3 d = dobj - dp;
        float ux = st[kPadShearX] + dot(d, f.tx), uy = st[kPadShearY] + dot(d, f.ty);
        float kt = st[kPadShearStiffness];
        if (!(kt > 0.0f)) kt = m.default_stiffness / m.tangential_ratio;
        float ustar = 1.5f * mu * W / kt;
        float u = sqrtf(ux * ux + uy * uy);
        bool sliding = u >= ustar;
        if (sliding && u > 0.0f) {
            ux *= ustar / u;
            uy *= ustar / u;
            u = ustar;
        }
        st[kPadShearX] = ux;
        st[kPadShearY] = uy;
        if (!(u > 0.0f) || !(ustar > 0.0f)) continue;
        float x = 1.0f - u / ustar;
        float Q = mu * W * (1.0f - x * sqrtf(x));
        V3 dir = (f.tx * ux + f.ty * uy) * (1.0f / u);
        V3 J = dir * (Q * p.h);  // impulse on the pad, along the object's motion
        // Critical damping of the stuck gel: oppose the relative tangential
        // velocity, never reversing it.
        if (!sliding) {
            V3 vrel = body_vel(b, other) + cross(body_angvel(b, other), ro) - body_vel(b, g) -
                      cross(body_angvel(b, g), rp);
            V3 vt = vrel - f.nz * dot(vrel, f.nz);
            float vlen = length(vt);
            if (vlen > 1e-12f) {
                V3 t = vt * (1.0f / vlen);
                float w = generalized_inv_mass(p, b, g, body_quat(b, g), rp, t) +
                          generalized_inv_mass(p, b, other, body_quat(b, other), ro, t);
                if (w > 0.0f) {
                    float c = 2.0f * sqrtf(kt / w);
                    float jd = fminf(c * vlen * p.h, vlen / w);
                    J = J + t * jd;
                }
            }
        }
        apply_velocity_impulse(p, b, g, rp, J);
        apply_velocity_impulse(p, b, other, ro, -J);
        V3 F = J * (1.0f / p.h);
        st[kPadForce] = F.x;
        st[kPadForce + 1] = F.y;
        st[kPadForce + 2] = F.z;
    }
}


// Positional correction of size `magnitude` along dir (a moves +dir, b -dir).
// Returns the Lagrange multiplier increment, or 0 if neither body can move.
PHYS_HD float correct_positions(const Params& p, const Buffers& b, const Contact& k, V3 ra,
                                V3 rb, V3 dir, float magnitude) {
    float w = generalized_inv_mass(p, b, k.a, load4(b.quat, k.a), ra, dir) +
              generalized_inv_mass(p, b, k.b, load4(b.quat, k.b), rb, dir);
    if (w <= 0.0f) return 0.0f;
    float dlambda = magnitude / w;
    V3 P = dir * dlambda;
    apply_position_impulse(p, b, k.a, ra, P);
    apply_position_impulse(p, b, k.b, rb, -P);
    return dlambda;
}

PHYS_HD void solve_contact_position(const Params& p, const Buffers& b, Contact& k) {
    V3 n = {k.n[0], k.n[1], k.n[2]};
    V3 pa, pb, ra, rb;
    contact_points(b, k, pa, pb, ra, rb);
    float depth = dot(pb - pa, n);
    if (k.compliance > 0.0f) {
        // Compliant (XPBD): settles where depth = compliance * force.
        if (depth <= 0.0f && k.lambda_n <= 0.0f) return;
        float alpha = k.compliance / (p.h * p.h);
        float w = generalized_inv_mass(p, b, k.a, load4(b.quat, k.a), ra, n) +
                  generalized_inv_mass(p, b, k.b, load4(b.quat, k.b), rb, n);
        if (w + alpha <= 0.0f) return;
        float dl = fmaxf((depth - alpha * k.lambda_n) / (w + alpha), -k.lambda_n);  // never pull
        // The gel bottoms out on its backing: no deeper than the pad allows.
        float limit = k.pad >= 0 ? b.sensors[k.pad].max_indentation : 0.0f;
        float beyond = depth - w * dl - limit;
        if (limit > 0.0f && beyond > 0.0f) dl += beyond / w;
        V3 P = n * dl;
        apply_position_impulse(p, b, k.a, ra, P);
        apply_position_impulse(p, b, k.b, rb, -P);
        k.lambda_n += dl;
        return;
    }
    if (depth <= 0.0f) return;
    k.lambda_n += correct_positions(p, b, k, ra, rb, n, depth);
}

PHYS_HD void solve_contact_static_friction(const Params& p, const Buffers& b, Contact& k) {
    if (k.lambda_n <= 0.0f || k.compliance > 0.0f) return;  // coupled pads: solve_pad_shear
    V3 n = {k.n[0], k.n[1], k.n[2]};
    V3 pa, pb, ra, rb;

    // Static friction: undo the tangential slip of the contact points over
    // this substep, as long as the total correction stays inside the
    // friction cone. The correction is accumulated as a vector so that
    // back-and-forth adjustments across iterations don't use up the cone.
    contact_points(b, k, pa, pb, ra, rb);
    V3 slip = (load3(b.disp, k.a) + cross(load3(b.drot, k.a), ra)) -
              (load3(b.disp, k.b) + cross(load3(b.drot, k.b), rb));
    V3 slip_t = slip - n * dot(slip, n);
    float len = length(slip_t);
    if (len <= 1e-20f) return;
    V3 dir = slip_t * (-1.0f / len);
    float w = generalized_inv_mass(p, b, k.a, load4(b.quat, k.a), ra, dir) +
              generalized_inv_mass(p, b, k.b, load4(b.quat, k.b), rb, dir);
    if (w <= 0.0f) return;
    V3 old_total = {k.static_friction[0], k.static_friction[1], k.static_friction[2]};
    V3 total = old_total + dir * (len / w);
    float limit = k.mu * k.lambda_n;
    if (dot(total, total) > limit * limit) return;  // slipping: dynamic friction applies
    V3 P = total - old_total;
    apply_position_impulse(p, b, k.a, ra, P);
    apply_position_impulse(p, b, k.b, rb, -P);
    k.static_friction[0] = total.x;
    k.static_friction[1] = total.y;
    k.static_friction[2] = total.z;
}



// Velocity-level dynamic friction for one contact. The normal impulse over
// the substep is lambda_n / h plus the restitution impulse, so the friction
// impulse is capped at mu times that. It accumulates across iterations, with the total kept
// inside the friction cone.
PHYS_HD void solve_contact_friction(const Params& p, const Buffers& b, Contact& k) {
    if (k.lambda_n <= 0.0f || k.compliance > 0.0f) return;  // not touching / coupled pad
    V3 n = {k.n[0], k.n[1], k.n[2]};
    V3 pa, pb, ra, rb;
    contact_points(b, k, pa, pb, ra, rb);
    V3 v = relative_velocity(b, k, ra, rb);
    V3 vt = v - n * dot(v, n);
    float vt_len = length(vt);
    if (vt_len <= 1e-9f) return;
    V3 t = vt * (1.0f / vt_len);
    float w = generalized_inv_mass(p, b, k.a, load4(b.quat, k.a), ra, t) +
              generalized_inv_mass(p, b, k.b, load4(b.quat, k.b), rb, t);
    if (w <= 0.0f) return;
    V3 old_total = {k.friction_impulse[0], k.friction_impulse[1], k.friction_impulse[2]};
    V3 total = old_total + t * (-vt_len / w);
    // The normal impulse is the position solve's push (lambda_n / h) plus the
    // restitution solve's correction, which is negative when it removes the
    // separating velocity the push left behind.
    float limit = k.mu * fmaxf(k.lambda_n / p.h + k.normal_impulse, 0.0f);
    float total_len = length(total);
    if (total_len > limit) total = total * (limit / total_len);
    apply_contact_impulse(p, b, k, ra, rb, total - old_total);
    k.friction_impulse[0] = total.x;
    k.friction_impulse[1] = total.y;
    k.friction_impulse[2] = total.z;
}

// Velocity-level restitution for one contact: sets the normal relative
// velocity to -e times the approach speed. Slow contacts don't bounce, so
// resting bodies settle.
PHYS_HD void solve_contact_restitution(const Params& p, const Buffers& b, Contact& k) {
    if (k.lambda_n <= 0.0f || k.compliance > 0.0f) return;  // coupled pads: damped in solve_pad_shear
    V3 n = {k.n[0], k.n[1], k.n[2]};
    V3 pa, pb, ra, rb;
    contact_points(b, k, pa, pb, ra, rb);
    float vn = dot(relative_velocity(b, k, ra, rb), n);
    float e = fabsf(k.vn_pre) <= p.rest_threshold ? 0.0f : k.e;
    float dvn = fmaxf(-e * k.vn_pre, 0.0f) - vn;
    float w = generalized_inv_mass(p, b, k.a, load4(b.quat, k.a), ra, n) +
              generalized_inv_mass(p, b, k.b, load4(b.quat, k.b), rb, n);
    if (w > 0.0f) {
        apply_contact_impulse(p, b, k, ra, rb, n * (dvn / w));
        k.normal_impulse += dvn / w;
    }
}

PHYS_HD void rigid_solve_positions(const Params& p, const Buffers& b, int env) {
    ContactSink sink{p, b, b.contacts + (size_t)env * p.max_contacts, 0};
    int base = env * p.nbody;
    for (int i = base; i < base + p.nbody; i++) {
        if (!b.enabled[i]) continue;
        for (int j = i + 1; j < base + p.nbody; j++)
            if (b.enabled[j]) collide_pair(sink, i, j);
    }
    b.ncontact[env] = sink.count;
    int n = sink.count < p.max_contacts ? sink.count : p.max_contacts;
    if (p.nsensor > 0) tag_pad_contacts(p, b, env, sink.out, n);
    // Joints and penetration are resolved first, over several symmetric
    // Gauss-Seidel sweeps (alternating direction, so the fixed order does not
    // bias resting bodies into creeping). Only then is static friction
    // applied, in a single sweep: correcting one corner of a box rotates it
    // and slides its other corners, but those slides cancel once every
    // corner has been corrected. Interleaving friction with the normal
    // iterations makes it fight those transient slides, which can exhaust
    // the friction cone of a resting box and makes tall stacks rock. A final
    // sweep removes the joint error and penetration friction introduced.
    for (int j = 0; j < p.njoint; j++) b.joint_lambda[env * p.njoint + j] = 0.0f;
    for (int it = 0; it < p.position_iterations; it++) {
        for (int j = 0; j < p.njoint; j++) {
            solve_joint_position(p, b, env, (it & 1) ? p.njoint - 1 - j : j);
        }
        for (int k = 0; k < n; k++) {
            solve_contact_position(p, b, sink.out[(it & 1) ? n - 1 - k : k]);
        }
    }
    for (int k = 0; k < n; k++) solve_contact_static_friction(p, b, sink.out[k]);
    for (int j = p.njoint - 1; j >= 0; j--) solve_joint_position(p, b, env, j);
    for (int k = n - 1; k >= 0; k--) solve_contact_position(p, b, sink.out[k]);
}

// Restitution is solved first, over several sweeps, so that an impact on
// several corners at once lifts the body evenly instead of twisting it.
// Friction and restitution sweeps then alternate, so friction sees the
// correct normal velocities and restitution can undo the small normal
// motion that friction impulses (applied below the centre of mass) cause.
// Sweeps alternate direction (symmetric Gauss-Seidel).
PHYS_HD void rigid_solve_velocities(const Params& p, const Buffers& b, int env) {
    Contact* contacts = b.contacts + (size_t)env * p.max_contacts;
    int n = b.ncontact[env] < p.max_contacts ? b.ncontact[env] : p.max_contacts;
    for (int j = 0; j < p.njoint; j++) solve_joint_velocity(p, b, env, j);
    for (int it = 0; it < p.velocity_iterations; it++)
        for (int k = 0; k < n; k++)
            solve_contact_restitution(p, b, contacts[(it & 1) ? n - 1 - k : k]);
    for (int it = 0; it < p.velocity_iterations; it++) {
        for (int k = 0; k < n; k++)
            solve_contact_friction(p, b, contacts[(it & 1) ? n - 1 - k : k]);
        for (int k = 0; k < n; k++)
            solve_contact_restitution(p, b, contacts[(it & 1) ? k : n - 1 - k]);
    }
    // Last, so that restitution (which zeroes the normal velocity of every
    // active contact, e.g. an object resting on a table) cannot cancel the
    // gel's pull on an object being lifted.
    if (p.nsensor > 0) solve_pad_shear(p, b, env);
}

}  // namespace detail
}  // namespace phys
