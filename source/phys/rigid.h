#pragma once

// Rigid solver: per-body and per-env routines shared by the CPU and CUDA
// backends. Bodies rotate, have a shape (sphere, capsule, box, plane) and
// friction. Each step is split into substeps; each substep runs
//
//   rigid_integrate          per body  predict position and orientation
//   rigid_solve_positions    per env   detect contacts, then push bodies
//                                      apart and apply static friction
//   rigid_update_velocities  per body  velocities from the position change
//   rigid_solve_velocities   per env   dynamic friction and restitution
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

PHYS_HD bool is_dynamic(const Params& p, const Buffers& b, int g) {
    return b.inv_mass[g % p.nbody] > 0.0f;
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
    float im = b.inv_mass[g % p.nbody];
    if (im == 0.0f) return 0.0f;
    V3 rn = cross(r, dir);
    return im + dot(rn, inv_inertia_times(p, b, g, q, rn));
}

// Positional impulse P applied at offset r.
PHYS_HD void apply_position_impulse(const Params& p, const Buffers& b, int g, V3 r, V3 P) {
    float im = b.inv_mass[g % p.nbody];
    if (im == 0.0f) return;
    store3(b.pos, g, load3(b.pos, g) + P * im);
    Q4 q = load4(b.quat, g);
    store4(b.quat, g, rotate_by(q, inv_inertia_times(p, b, g, q, cross(r, P))));
}

// Velocity impulse P applied at offset r.
PHYS_HD void apply_velocity_impulse(const Params& p, const Buffers& b, int g, V3 r, V3 P) {
    float im = b.inv_mass[g % p.nbody];
    if (im == 0.0f) return;
    store3(b.vel, g, load3(b.vel, g) + P * im);
    Q4 q = load4(b.quat, g);
    store3(b.angvel, g, load3(b.angvel, g) + inv_inertia_times(p, b, g, q, cross(r, P)));
}

// --- per-body phases ---------------------------------------------------------

PHYS_HD void rigid_integrate(const Params& p, const Buffers& b, int g) {
    V3 x = load3(b.pos, g);
    Q4 q = load4(b.quat, g);
    store3(b.prev_pos, g, x);
    store4(b.prev_quat, g, q);
    if (!b.enabled[g] || !is_dynamic(p, b, g)) return;

    V3 grav = {p.gravity[0], p.gravity[1], p.gravity[2]};
    V3 v = load3(b.vel, g) + grav * p.h;
    store3(b.vel, g, v);
    store3(b.pos, g, x + v * p.h);

    // Torque-free rotation, including the gyroscopic term, in the body frame.
    V3 ii = model3(b.inv_inertia, g % p.nbody);
    V3 wb = inv_rotate(q, load3(b.angvel, g));
    V3 Iw = {wb.x / ii.x, wb.y / ii.y, wb.z / ii.z};
    wb = wb + mul(ii, cross(Iw, wb)) * p.h;
    V3 w = rotate(q, wb);
    store3(b.angvel, g, w);
    store4(b.quat, g, rotate_by(q, w * p.h));
}

PHYS_HD void rigid_update_velocities(const Params& p, const Buffers& b, int g) {
    if (!b.enabled[g] || !is_dynamic(p, b, g)) return;
    float inv_h = 1.0f / p.h;
    store3(b.vel, g, (load3(b.pos, g) - load3(b.prev_pos, g)) * inv_h);
    Q4 dq = qmul(load4(b.quat, g), conj(load4(b.prev_quat, g)));
    V3 w = V3{dq.x, dq.y, dq.z} * (2.0f * inv_h);
    store3(b.angvel, g, dq.w >= 0.0f ? w : -w);
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
        V3 xa = load3(b.pos, a), xb = load3(b.pos, c);
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
    V3 x = load3(b.pos, g);
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
    box.c = load3(b.pos, g);
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
    if (!is_dynamic(p, b, i) && !is_dynamic(p, b, j)) return;
    if (si != kPlane && sj != kPlane) {
        V3 d = load3(b.pos, i) - load3(b.pos, j);
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
        V3 ca = load3(b.pos, a);
        if (sj == kSphere) {
            spheres(s, a, ca, ra, c, load3(b.pos, c), rc);
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

// --- per-env phases ----------------------------------------------------------

PHYS_HD void contact_points(const Buffers& b, const Contact& k, V3& pa, V3& pb, V3& ra,
                            V3& rb) {
    ra = rotate(load4(b.quat, k.a), V3{k.ra[0], k.ra[1], k.ra[2]});
    rb = rotate(load4(b.quat, k.b), V3{k.rb[0], k.rb[1], k.rb[2]});
    pa = load3(b.pos, k.a) + ra;
    pb = load3(b.pos, k.b) + rb;
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
    if (depth <= 0.0f) return;
    k.lambda_n += correct_positions(p, b, k, ra, rb, n, depth);
}

PHYS_HD void solve_contact_static_friction(const Params& p, const Buffers& b, Contact& k) {
    if (k.lambda_n <= 0.0f) return;
    V3 n = {k.n[0], k.n[1], k.n[2]};
    V3 pa, pb, ra, rb;

    // Static friction: undo the tangential slip of the contact points over
    // this substep, as long as the total correction stays inside the
    // friction cone. The correction is accumulated as a vector so that
    // back-and-forth adjustments across iterations don't use up the cone.
    contact_points(b, k, pa, pb, ra, rb);
    V3 xa0 = load3(b.prev_pos, k.a), xb0 = load3(b.prev_pos, k.b);
    V3 pa0 = xa0 + rotate(load4(b.prev_quat, k.a), V3{k.ra[0], k.ra[1], k.ra[2]});
    V3 pb0 = xb0 + rotate(load4(b.prev_quat, k.b), V3{k.rb[0], k.rb[1], k.rb[2]});
    V3 slip = (pa - pa0) - (pb - pb0);
    V3 slip_t = slip - n * dot(slip, n);
    float len = length(slip_t);
    if (len <= 1e-9f) return;
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

// Velocity-level dynamic friction for one contact. The normal impulse over
// the substep is lambda_n / h, so the friction impulse is capped at
// mu * lambda_n / h. It accumulates across iterations, with the total kept
// inside the friction cone.
PHYS_HD void solve_contact_friction(const Params& p, const Buffers& b, Contact& k) {
    if (k.lambda_n <= 0.0f) return;  // not touching this substep
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
    float limit = k.mu * k.lambda_n / p.h;
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
PHYS_HD void solve_contact_restitution(const Params& p, const Buffers& b, const Contact& k) {
    if (k.lambda_n <= 0.0f) return;
    V3 n = {k.n[0], k.n[1], k.n[2]};
    V3 pa, pb, ra, rb;
    contact_points(b, k, pa, pb, ra, rb);
    float vn = dot(relative_velocity(b, k, ra, rb), n);
    float e = fabsf(k.vn_pre) <= p.rest_threshold ? 0.0f : k.e;
    float dvn = fmaxf(-e * k.vn_pre, 0.0f) - vn;
    float w = generalized_inv_mass(p, b, k.a, load4(b.quat, k.a), ra, n) +
              generalized_inv_mass(p, b, k.b, load4(b.quat, k.b), rb, n);
    if (w > 0.0f) apply_contact_impulse(p, b, k, ra, rb, n * (dvn / w));
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
    // Penetration is resolved first, over several symmetric Gauss-Seidel
    // sweeps (alternating direction, so the fixed contact order does not
    // bias resting bodies into creeping). Only then is static friction
    // applied, in a single sweep: correcting one corner of a box rotates it
    // and slides its other corners, but those slides cancel once every
    // corner has been corrected. Interleaving friction with the normal
    // iterations makes it fight those transient slides, which can exhaust
    // the friction cone of a resting box and makes tall stacks rock. A final
    // normal sweep removes any penetration the friction pass introduced.
    for (int it = 0; it < p.position_iterations; it++)
        for (int k = 0; k < n; k++)
            solve_contact_position(p, b, sink.out[(it & 1) ? n - 1 - k : k]);
    for (int k = 0; k < n; k++) solve_contact_static_friction(p, b, sink.out[k]);
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
    for (int it = 0; it < p.velocity_iterations; it++)
        for (int k = 0; k < n; k++)
            solve_contact_restitution(p, b, contacts[(it & 1) ? n - 1 - k : k]);
    for (int it = 0; it < p.velocity_iterations; it++) {
        for (int k = 0; k < n; k++)
            solve_contact_friction(p, b, contacts[(it & 1) ? n - 1 - k : k]);
        for (int k = 0; k < n; k++)
            solve_contact_restitution(p, b, contacts[(it & 1) ? k : n - 1 - k]);
    }
}

}  // namespace detail
}  // namespace phys
