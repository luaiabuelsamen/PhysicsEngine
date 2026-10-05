#pragma once

// Tactile sensor pads on rigid bodies (see phys::TactileSensorDesc).
//
// The rigid solver decides how bodies move and how hard they press: the
// force each pad carries is the contact force on the pad's face, averaged
// over the substeps of a step (a single substep's contact force can
// alternate between zero and twice the mean as Gauss-Seidel settles a
// resting contact), and the touching bodies' poses are those at the end of
// the step. The tactile model then decides how that force
// is distributed over the pad's gel. The gel is a linear elastic half-space
// sampled on the pad's grid, as in ElasticPatch (tactile.h), and the
// indenter's shape comes from ray casts against the touching bodies.
//
// Normal: the load-controlled contact problem
//     find P >= 0 with sum P = W, and an approach delta, such that
//     h_i + (K P)_i = delta where P_i > 0 and >= delta elsewhere,
// where h is the gap to the indenter and K the half-space influence matrix,
// solved with the conjugate gradient method of Polonsky & Keer (1999,
// Wear 231:206).
//
// Tangential: for a shear force Q, q = mu (P - P*) Q / |Q|, where P* solves
// the same normal problem at the reduced load W - |Q| / mu (Ciavarella 1998,
// Jaeger 1998). The cells where P* > 0 stick and the rest of the contact
// slips. This is the exact partial-slip solution for a monotonically applied
// shear when the tangential and normal kernels are proportional, as they are
// in the angle-averaged model used here. It keeps no per-cell slip history,
// so it does not model hysteresis under reversed or cyclic shear; the
// standalone ElasticPatch does.
//
// The force is accumulated by one thread per (env, sensor) every substep.
// The contact solve after the step runs one CUDA block per (env, sensor),
// with threads over cells; the CPU backend runs the identical sequence of
// floating-point operations serially (SerialExec below), so the two
// backends agree bit for bit.

#include "phys/common.h"
#include "phys/rigid.h"

namespace phys {
namespace detail {

// Output channels of every cell.
enum TactileChannel : int {
    kPressure = 0,      // Pa
    kShearX = 1,        // Pa, along the pad's x axis
    kShearY = 2,
    kDeflection = 3,    // normal deflection of the gel surface (m), into the pad
    kDisplacementX = 4, // tangential displacement of the gel surface (m)
    kDisplacementY = 5,
    kStick = 6,         // 1 where the contact sticks, else 0
};

constexpr float kTactileMiss = 1e30f;  // gap of cells that see no indenter

// --- geometry ----------------------------------------------------------------

// Distance t >= 0 along the ray o + t d (d unit) to where it first enters
// body g's shape; 0 if o is inside; kTactileMiss if the ray misses.
PHYS_HD float ray_enter(const Params& p, const Buffers& b, int g, V3 o, V3 d) {
    int body = g % p.nbody;
    Q4 q = load4(b.quat, g);
    V3 lo = inv_rotate(q, o - load3(b.pos, g));
    V3 ld = inv_rotate(q, d);
    V3 size = model3(b.size, body);
    const float miss = kTactileMiss;
    auto sphere = [&](V3 c, float r) -> float {
        V3 oc = lo - c;
        float bb = dot(oc, ld), cc = dot(oc, oc) - r * r;
        if (cc <= 0.0f) return 0.0f;
        float disc = bb * bb - cc;
        if (bb > 0.0f || disc < 0.0f) return miss;
        return -bb - sqrtf(disc);
    };
    switch (b.shape[body]) {
        case kSphere: return sphere(V3{0.0f, 0.0f, 0.0f}, size.x);
        case kCapsule: {
            float r = size.x, hl = size.y;
            float cy = fminf(fmaxf(lo.y, -hl), hl);
            V3 off = {lo.x, lo.y - cy, lo.z};
            if (dot(off, off) <= r * r) return 0.0f;
            float t = fminf(sphere(V3{0.0f, hl, 0.0f}, r), sphere(V3{0.0f, -hl, 0.0f}, r));
            float a = ld.x * ld.x + ld.z * ld.z;
            if (a > 1e-12f) {
                float bb = lo.x * ld.x + lo.z * ld.z;
                float cc = lo.x * lo.x + lo.z * lo.z - r * r;
                float disc = bb * bb - a * cc;
                if (disc >= 0.0f) {
                    float tc = (-bb - sqrtf(disc)) / a;
                    float y = lo.y + tc * ld.y;
                    if (tc >= 0.0f && y >= -hl && y <= hl) t = fminf(t, tc);
                }
            }
            return t;
        }
        case kBox: {
            float tmin = 0.0f, tmax = miss;
            for (int k = 0; k < 3; k++) {
                float ok = get(lo, k), dk = get(ld, k), e = get(size, k);
                if (fabsf(dk) < 1e-12f) {
                    if (ok < -e || ok > e) return miss;
                    continue;
                }
                float t1 = (-e - ok) / dk, t2 = (e - ok) / dk;
                if (t1 > t2) { float s = t1; t1 = t2; t2 = s; }
                tmin = fmaxf(tmin, t1);
                tmax = fminf(tmax, t2);
            }
            return tmin <= tmax ? tmin : miss;
        }
        case kPlane: {
            // Solid on the local -y side.
            if (lo.y <= 0.0f) return 0.0f;
            if (ld.y >= 0.0f) return miss;
            return -lo.y / ld.y;
        }
        default: return miss;
    }
}

// --- executors -----------------------------------------------------------------

// Threads per (env, sensor) block on the GPU: T = Params::tactile_threads,
// one to four warps depending on the sensor size. Reductions are defined the
// same way on both backends, so they round identically: thread t sums
// elements t, t + T, t + 2T, ... in order; within each warp the 32 partial
// sums are added pairwise at distances 16, 8, 4, 2, 1; the warp sums are
// then added in order.
constexpr int kTactileMaxThreads = 128;
constexpr int kWarp = 32;

// Threads for sensors of up to `cells` cells: about 8 cells per thread.
inline int tactile_threads_for(int cells) {
    int t = kWarp;
    while (t < kTactileMaxThreads && t * 8 < cells) t *= 2;
    return t;
}

// Runs a block's work on one CPU thread. `list` and `vals` hold n entries,
// `red` kTactileMaxThreads.
//
// Lists of cells hold packed grid coordinates, (iy << 16) | ix, so that the
// inner loops of the influence sums need no integer division.
struct SerialExec {
    int threads;  // T, emulated
    int* list;
    float* vals;
    float* red;
    const float* kernel;  // influence coefficients of the sensor being solved
    float* work = nullptr;  // working arrays, if not in global scratch
    int* cand = nullptr;
    void load_kernel(const float* table, int) { kernel = table; }

    template <class F>
    void for_each(int n, F f) {
        for (int i = 0; i < n; i++) f(i);
    }
    template <class F>
    float sum(int n, F f) {
        for (int t = 0; t < threads; t++) {
            float s = 0.0f;
            for (int i = t; i < n; i += threads) s += f(i);
            red[t] = s;
        }
        for (int w = 0; w < threads / kWarp; w++) {
            float* v = red + w * kWarp;
            for (int off = kWarp / 2; off > 0; off >>= 1)
                for (int l = 0; l < off; l++) v[l] += v[l + off];
        }
        float total = red[0];
        for (int w = 1; w < threads / kWarp; w++) total += red[w * kWarp];
        return total;
    }
    // Writes value(k) for every k < n with keep(k), in increasing k; returns the count.
    template <class Keep, class Value>
    int compact(int n, Keep keep, Value value, int* out) {
        int count = 0;
        for (int k = 0; k < n; k++)
            if (keep(k)) out[count++] = value(k);
        return count;
    }
};

#ifdef __CUDACC__
// Runs a block's work on Params::tactile_threads GPU threads; the arrays are in
// shared memory. Every call ends with a barrier, so writes made in one call
// are visible to the next.
struct BlockExec {
    int* list;
    float* vals;
    float* red;     // one per warp
    int* counts;    // one per warp
    float* kernel;  // shared copy of the sensor's influence coefficients
    float* work;    // shared working arrays, or nullptr to use global scratch
    int* cand;

    __device__ void load_kernel(const float* table, int n) {
        for (int i = threadIdx.x; i < n; i += (int)blockDim.x) kernel[i] = table[i];
        __syncthreads();
    }
    template <class F>
    __device__ void for_each(int n, F f) {
        for (int i = threadIdx.x; i < n; i += (int)blockDim.x) f(i);
        __syncthreads();
    }
    template <class F>
    __device__ float sum(int n, F f) {
        float v = 0.0f;
        for (int i = threadIdx.x; i < n; i += (int)blockDim.x) v += f(i);
        for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if ((threadIdx.x & (kWarp - 1)) == 0) red[threadIdx.x / kWarp] = v;
        __syncthreads();
        float total = red[0];
        for (int w = 1; w < (int)(blockDim.x / kWarp); w++) total += red[w];
        __syncthreads();
        return total;
    }
    template <class Keep, class Value>
    __device__ int compact(int n, Keep keep, Value value, int* out) {
        // Each thread takes a contiguous chunk, so the output is in order.
        int chunk = (n + (int)blockDim.x - 1) / (int)blockDim.x;
        int lo = threadIdx.x * chunk, hi = min(n, lo + chunk);
        int mine = 0;
        for (int k = lo; k < hi; k++) mine += keep(k) ? 1 : 0;
        // Exclusive prefix sum of the counts across the block.
        int lane = threadIdx.x & (kWarp - 1), warp = threadIdx.x / kWarp;
        int incl = mine;
        for (int off = 1; off < kWarp; off <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= off) incl += up;
        }
        if (lane == kWarp - 1) counts[warp] = incl;
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < (int)(blockDim.x / kWarp); w++) {
            if (w < warp) before += counts[w];
            total += counts[w];
        }
        int at = before + incl - mine;
        for (int k = lo; k < hi; k++)
            if (keep(k)) out[at++] = value(k);
        __syncthreads();
        return total;
    }
};
#endif

// --- the contact solve -------------------------------------------------------

// Float scratch arrays, n entries each, per (env, sensor).
// kWarm keeps the last step's cell forces, to warm-start the next solve.
enum { kGap = 0, kLoad = 1, kLoadPrev = 2, kDir = 3, kResid = 4, kKDir = 5, kWarm = 6, kTactileArrays = 7 };

PHYS_HD int pack_cell(int c, int nx) { return ((c / nx) << 16) | (c % nx); }
PHYS_HD int unpack_cell(int packed, int nx) { return (packed >> 16) * nx + (packed & 0xffff); }

struct TactileProblem {
    const TactileModel* s;
    const float* kernel;  // [ny * nx] influence coefficients (global)
    float* work;          // kWarm working arrays of n (kGap .. kKDir)
    float* warm;          // the last step's cell forces (global, persistent)
    int* candidates;      // cells that see an indenter (cell indices)
    int n;

    PHYS_HD float* array(int a) const { return a == kWarm ? warm : work + (long long)a * n; }
};

// Influence of a load on the cell at packed coordinates `pj` on the cell at
// (ix, iy).
PHYS_HD float influence(const float* kernel, int nx, int ix, int iy, int pj) {
    int dx = ix - (pj & 0xffff), dy = iy - (pj >> 16);
    return kernel[(dy < 0 ? -dy : dy) * nx + (dx < 0 ? -dx : dx)];
}

PHYS_HD TactileProblem tactile_problem(const Params& p, const Buffers& b, int env, int sensor) {
    const TactileModel& s = b.sensors[sensor];
    int n = s.nx * s.ny;
    TactileProblem pr;
    pr.s = &s;
    pr.kernel = b.tactile_kernel + s.kernel_offset;
    pr.work = b.tactile_scratch + s.scratch_offset + (long long)env * kTactileArrays * n;
    pr.warm = pr.work + (long long)kWarm * n;
    pr.candidates = b.tactile_list + s.list_offset + (long long)env * n;
    pr.n = n;
    return pr;
}

// Polonsky-Keer: cell forces P (array kLoad) summing to W over the m
// candidate cells, given their gaps (array kGap). With `warm`, starts from
// the forces already in kLoad (rescaled), else from a uniform load. Returns
// the iterations used. The contact set lives in ex.list, its values in
// ex.vals.
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <class Exec>
PHYS_HD int solve_normal_load(Exec& ex, const TactileProblem& pr, int m, float W, bool warm) {
    const int nx = pr.s->nx;
    const int* cand = pr.candidates;
    const float* K = ex.kernel;
    float* P = pr.array(kLoad);
    float* Pprev = pr.array(kLoadPrev);
    float* D = pr.array(kDir);
    float* R = pr.array(kResid);
    float* KD = pr.array(kKDir);
    const float* gap = pr.array(kGap);
    int* list = ex.list;
    float* vals = ex.vals;
    float total0 = warm ? ex.sum(m, [&](int k) { return fmaxf(P[cand[k]], 0.0f); }) : 0.0f;
    float start = W / (float)m, rescale = total0 > 0.0f ? W / total0 : 0.0f;
    ex.for_each(m, [&](int k) {
        int i = cand[k];
        P[i] = total0 > 0.0f ? fmaxf(P[i], 0.0f) * rescale : start;
        D[i] = 0.0f;
    });
    float G_old = 1.0f;
    bool conjugate = false;
    int it = 0;
    for (; it < pr.s->max_iterations; it++) {
        int na = ex.compact(m, [&](int k) { return P[cand[k]] > 0.0f; },
                            [&](int k) { return pack_cell(cand[k], nx); }, list);
        if (na == 0) break;
        // Surface gaps h + K P, centred on their mean over the contact (the
        // approach). The deflection K P is never negative, so a cell whose gap
        // alone exceeds the approach cannot be penetrated; its deflection is
        // not needed, only the sign of its residual.
        ex.for_each(na, [&](int a) { vals[a] = P[unpack_cell(list[a], nx)]; });
        ex.for_each(na, [&](int a) {
            int pi = list[a], ix = pi & 0xffff, iy = pi >> 16;
            float u = 0.0f;
            for (int c = 0; c < na; c++) u += influence(K, nx, ix, iy, list[c]) * vals[c];
            R[iy * nx + ix] = gap[iy * nx + ix] + u;
        });
        float gbar = ex.sum(na, [&](int a) { return R[unpack_cell(list[a], nx)]; }) / (float)na;
        ex.for_each(m, [&](int k) {
            int i = cand[k];
            if (P[i] > 0.0f) {
                R[i] -= gbar;
            } else if (gap[i] < gbar) {
                int ix = i % nx, iy = i / nx;
                float u = 0.0f;
                for (int a = 0; a < na; a++) u += influence(K, nx, ix, iy, list[a]) * vals[a];
                R[i] = gap[i] + u - gbar;
            } else {
                R[i] = gap[i] - gbar;  // >= 0: stays out of contact
            }
        });
        float G = ex.sum(na, [&](int a) {
            float g = R[unpack_cell(list[a], nx)];
            return g * g;
        });
        // Conjugate direction on the contact, zero elsewhere.
        float beta = conjugate ? G / G_old : 0.0f;
        ex.for_each(m, [&](int k) {
            int i = cand[k];
            D[i] = P[i] > 0.0f ? R[i] + beta * D[i] : 0.0f;
        });
        G_old = G;
        ex.for_each(na, [&](int a) { vals[a] = D[unpack_cell(list[a], nx)]; });
        ex.for_each(na, [&](int a) {
            int pi = list[a], ix = pi & 0xffff, iy = pi >> 16;
            float r = 0.0f;
            for (int c = 0; c < na; c++) r += influence(K, nx, ix, iy, list[c]) * vals[c];
            KD[iy * nx + ix] = r;
        });
        float rbar = ex.sum(na, [&](int a) { return KD[unpack_cell(list[a], nx)]; }) / (float)na;
        float num = ex.sum(na, [&](int a) {
            int i = unpack_cell(list[a], nx);
            return R[i] * D[i];
        });
        float den = ex.sum(na, [&](int a) {
            int i = unpack_cell(list[a], nx);
            return (KD[i] - rbar) * D[i];
        });
        if (!(den > 0.0f)) break;  // converged to round-off
        float tau = num / den;
        // Step, keep P >= 0, and load cells that the indenter penetrates.
        ex.for_each(m, [&](int k) { Pprev[cand[k]] = P[cand[k]]; });
        ex.for_each(na, [&](int a) {
            int i = unpack_cell(list[a], nx);
            P[i] = fmaxf(P[i] - tau * D[i], 0.0f);
        });
        float overlapping = ex.sum(m, [&](int k) {
            int i = cand[k];
            if (P[i] == 0.0f && R[i] < 0.0f) {
                P[i] = -tau * R[i];
                return 1.0f;
            }
            return 0.0f;
        });
        conjugate = overlapping == 0.0f;
        float total = ex.sum(m, [&](int k) { return P[cand[k]]; });
        if (!(total > 0.0f)) break;
        float scale = W / total;
        float change = ex.sum(m, [&](int k) {
            int i = cand[k];
            float v = P[i] * scale;
            P[i] = v;
            return fabsf(v - Pprev[i]);
        });
        if (change <= pr.s->tolerance * W) {
            it++;
            break;
        }
    }
    return it;
}

// Pad frame in world coordinates.
struct PadFrame {
    V3 origin, tx, ty, nz;
};

PHYS_HD PadFrame pad_frame(const Buffers& b, const TactileModel& s, int g) {
    Q4 qb = load4(b.quat, g);
    Q4 q = qmul(qb, Q4{s.frame[0], s.frame[1], s.frame[2], s.frame[3]});
    PadFrame f;
    f.origin = load3(b.pos, g) + rotate(qb, V3{s.origin[0], s.origin[1], s.origin[2]});
    f.tx = rotate(q, V3{1.0f, 0.0f, 0.0f});
    f.ty = rotate(q, V3{0.0f, 1.0f, 0.0f});
    f.nz = rotate(q, V3{0.0f, 0.0f, 1.0f});
    return f;
}

PHYS_HD float cell_local_x(const TactileModel& s, int c) {
    return -s.half_w + ((float)(c % s.nx) + 0.5f) * s.cell_x;
}
PHYS_HD float cell_local_y(const TactileModel& s, int c) {
    return -s.half_h + ((float)(c / s.nx) + 0.5f) * s.cell_y;
}

// Add the contact force on sensor `sensor`'s pad in env `env` from the
// current substep's contacts; `first` starts a new step.
PHYS_HD void tactile_accumulate(const Params& p, const Buffers& b, int env, int sensor, bool first) {
    const TactileModel& s = b.sensors[sensor];
    long long slot = (long long)env * p.nsensor + sensor;
    float* acc = b.tactile_accum + slot * kTactileAccum;
    int* touch = b.tactile_touch + slot * kTactileMaxTouching;
    if (first) {
        for (int k = 0; k < kTactileAccum; k++) acc[k] = 0.0f;
        for (int k = 0; k < kTactileMaxTouching; k++) touch[k] = -1;
    }
    const int g = env * p.nbody + s.body;
    if (!b.enabled[g]) return;
    PadFrame f = pad_frame(b, s, g);
    const Contact* contacts = b.contacts + (long long)env * p.max_contacts;
    int ncontact = b.ncontact[env] < p.max_contacts ? b.ncontact[env] : p.max_contacts;
    float inv_h = 1.0f / p.h, inv_h2 = inv_h * inv_h;
    float margin_x = s.half_w + s.cell_x, margin_y = s.half_h + s.cell_y;
    for (int k = 0; k < ncontact; k++) {
        const Contact& c = contacts[k];
        if (c.lambda_n <= 0.0f || (c.a != g && c.b != g)) continue;
        bool on_a = c.a == g;
        V3 n = {c.n[0], c.n[1], c.n[2]};
        V3 push = on_a ? n : -n;  // direction the normal force pushes the pad's body
        if (dot(push, f.nz) > -0.5f) continue;  // not on the pad's face
        V3 pa, pb, ra, rb;
        contact_points(b, c, pa, pb, ra, rb);
        V3 rel = (on_a ? pa : pb) - f.origin;
        if (fabsf(dot(rel, f.tx)) > margin_x || fabsf(dot(rel, f.ty)) > margin_y) continue;
        V3 on_first = n * (c.lambda_n * inv_h2 + c.normal_impulse * inv_h) +
                      V3{c.static_friction[0], c.static_friction[1], c.static_friction[2]} * inv_h2 +
                      V3{c.friction_impulse[0], c.friction_impulse[1], c.friction_impulse[2]} * inv_h;
        V3 F = on_a ? on_first : -on_first;
        acc[0] += F.x;
        acc[1] += F.y;
        acc[2] += F.z;
        acc[3] += c.mu * c.lambda_n;
        acc[4] += c.lambda_n;
        int other = on_a ? c.b : c.a;
        for (int t = 0; t < kTactileMaxTouching; t++) {
            if (touch[t] == other) break;
            if (touch[t] < 0) {
                touch[t] = other;
                break;
            }
        }
    }
}

// Update the outputs of sensor `sensor` in env `env` from the force
// accumulated over the step's substeps.
#ifdef __CUDACC__
#pragma nv_exec_check_disable
#endif
template <class Exec>
PHYS_HD void tactile_update(Exec& ex, const Params& p, const Buffers& b, int env, int sensor) {
    const TactileModel& s = b.sensors[sensor];
    const int n = s.nx * s.ny;
    float* out = b.tactile + ((long long)env * p.ntactile + s.cell_offset) * kTactileChannels;
    float* force = b.tactile_force + ((long long)env * p.nsensor + sensor) * 3;
    ex.for_each(n * kTactileChannels, [&](int k) { out[k] = 0.0f; });

    // Every thread computes the same loads.
    const int g = env * p.nbody + s.body;
    bool enabled = b.enabled[g] != 0;
    PadFrame f = pad_frame(b, s, g);
    long long slot = (long long)env * p.nsensor + sensor;
    const float* acc = b.tactile_accum + slot * kTactileAccum;
    const int* touching = b.tactile_touch + slot * kTactileMaxTouching;
    int ntouch = 0;
    while (ntouch < kTactileMaxTouching && touching[ntouch] >= 0) ntouch++;
    float inv_substeps = 1.0f / (float)p.substeps;
    V3 F = V3{acc[0], acc[1], acc[2]} * inv_substeps;
    float W = -dot(F, f.nz);
    float Qx = dot(F, f.tx), Qy = dot(F, f.ty);
    float Q = sqrtf(Qx * Qx + Qy * Qy);
    float mu = acc[4] > 0.0f ? acc[3] / acc[4] : 0.0f;
    bool loaded = enabled && ntouch > 0 && W > 0.0f;
    ex.for_each(1, [&](int) {
        force[0] = loaded ? Qx : 0.0f;
        force[1] = loaded ? Qy : 0.0f;
        force[2] = loaded ? W : 0.0f;
    });
    if (!loaded) {
        float* warm = tactile_problem(p, b, env, sensor).array(kWarm);
        ex.for_each(n, [&](int c) { warm[c] = 0.0f; });
        return;
    }

    TactileProblem pr = tactile_problem(p, b, env, sensor);
    if (ex.work) {
        pr.work = ex.work;
        pr.candidates = ex.cand;
    }
    const int nx = s.nx;
    float* gap = pr.array(kGap);
    float* P = pr.array(kLoad);
    float* D = pr.array(kDir);
    float* warm = pr.array(kWarm);
    int* list = ex.list;
    float* vals = ex.vals;
    ex.load_kernel(pr.kernel, n);
    const float* K = ex.kernel;

    // Gap from each cell of the undeformed gel to the touching bodies, along
    // the pad normal.
    float depth = s.half_w > s.half_h ? s.half_w : s.half_h;
    ex.for_each(n, [&](int c) {
        float x = cell_local_x(s, c), y = cell_local_y(s, c);
        V3 o = f.origin + f.tx * x + f.ty * y - f.nz * depth;
        float t = kTactileMiss;
        for (int k = 0; k < ntouch; k++) t = fminf(t, ray_enter(p, b, touching[k], o, f.nz));
        float dome = s.dome_radius > 0.0f ? (x * x + y * y) / (2.0f * s.dome_radius) : 0.0f;
        gap[c] = t >= kTactileMiss ? kTactileMiss : t - depth + dome;
    });
    int m = ex.compact(n, [&](int c) { return gap[c] < kTactileMiss; }, [&](int c) { return c; }, pr.candidates);
    if (m == 0) return;
    const int* cand = pr.candidates;

    // Pressure, warm-started from the last step's, and the gel's normal
    // deflection from it.
    ex.for_each(m, [&](int k) { P[cand[k]] = warm[cand[k]]; });
    solve_normal_load(ex, pr, m, W, true);
    ex.for_each(n, [&](int c) { warm[c] = 0.0f; });
    ex.for_each(m, [&](int k) {
        int c = cand[k];
        out[kPressure * n + c] = P[c];  // forces for now
        warm[c] = P[c];
    });
    int na = ex.compact(m, [&](int k) { return P[cand[k]] > 0.0f; },
                        [&](int k) { return pack_cell(cand[k], nx); }, list);
    ex.for_each(na, [&](int a) { vals[a] = P[unpack_cell(list[a], nx)]; });
    ex.for_each(n, [&](int c) {
        int ix = c % nx, iy = c / nx;
        float w = 0.0f;
        for (int a = 0; a < na; a++) w += influence(K, nx, ix, iy, list[a]) * vals[a];
        out[kDeflection * n + c] = w;
    });

    // Shear: the difference between the full load and a reduced one, which
    // starts from the full-load solution. Within 2% of the friction cone the
    // whole contact counts as sliding: that is the step-to-step scatter of
    // the rigid solver's friction force while it slides, and contacts whose
    // area does not shrink with load (edges, flat faces) would otherwise
    // read as fully stuck right up to the cone.
    float reduced = mu > 0.0f ? W - Q / mu : 0.0f;
    if (Q == 0.0f) {
        // P* = P: everything sticks, no shear (P is still the full solution).
    } else if (reduced > 0.02f * W) {
        solve_normal_load(ex, pr, m, reduced, true);
    } else {
        ex.for_each(m, [&](int k) { P[cand[k]] = 0.0f; });
    }
    float slip_total = ex.sum(m, [&](int k) {
        int c = cand[k];
        float d = fmaxf(out[kPressure * n + c] - P[c], 0.0f);
        D[c] = d;
        out[kStick * n + c] = P[c] > 0.0f ? 1.0f : 0.0f;
        return d;
    });
    float qscale = Q > 0.0f && slip_total > 0.0f ? 1.0f / slip_total : 0.0f;
    int nq = ex.compact(m, [&](int k) { return D[cand[k]] > 0.0f; },
                        [&](int k) { return pack_cell(cand[k], nx); }, list);
    ex.for_each(nq, [&](int a) {
        int c = unpack_cell(list[a], nx);
        float frac = D[c] * qscale;  // share of the shear on this cell
        vals[a] = frac;
        out[kShearX * n + c] = frac * Qx;
        out[kShearY * n + c] = frac * Qy;
    });
    float ratio = s.tangential_ratio;
    ex.for_each(n, [&](int c) {
        int ix = c % nx, iy = c / nx;
        float v = 0.0f;
        for (int a = 0; a < nq; a++) v += influence(K, nx, ix, iy, list[a]) * vals[a];
        out[kDisplacementX * n + c] = v * Qx * ratio;
        out[kDisplacementY * n + c] = v * Qy * ratio;
    });
    const float inv_area = 1.0f / (s.cell_x * s.cell_y);
    ex.for_each(m, [&](int k) {
        int c = cand[k];
        out[kPressure * n + c] *= inv_area;
        out[kShearX * n + c] *= inv_area;
        out[kShearY * n + c] *= inv_area;
    });
}

}  // namespace detail
}  // namespace phys
