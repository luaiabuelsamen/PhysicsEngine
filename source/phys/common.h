#pragma once

// Types shared by the simulation routines (particle.h, rigid.h) and the
// backends that run them. Everything here is plain data so it can be passed
// to CUDA kernels by value.

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define PHYS_HD __host__ __device__ __forceinline__
#else
#define PHYS_HD inline
#endif

namespace phys {
namespace detail {

// Same values as phys::Shape.
enum ShapeType : int { kSphere = 0, kCapsule = 1, kBox = 2, kPlane = 3 };

struct Params {
    int nenv, nbody;
    bool rigid;  // Solver::Rigid, otherwise Solver::Particle
    float dt;
    float gravity[3];

    // Particle solver: walls and uniform grid broadphase.
    float lo[3], hi[3];
    float wall_restitution;
    float contact_correction;
    bool grid;
    float inv_cell;
    int dim[3];
    int ncell;         // cells per env
    int sentinel_key;  // key of disabled bodies; sorts after every real cell

    // Rigid solver.
    int substeps;
    int position_iterations;
    int velocity_iterations;
    float h;               // substep length, dt / substeps
    float rest_threshold;  // below this approach speed, contacts don't bounce
    int max_contacts;      // contact buffer capacity per env
};

// Fill in the fields that depend on the step length. Runs on the host, so
// both backends see identical values.
inline void set_step_length(Params& p, float dt) {
    p.dt = dt;
    p.h = dt / (float)p.substeps;
    float g = sqrtf(p.gravity[0] * p.gravity[0] + p.gravity[1] * p.gravity[1] +
                    p.gravity[2] * p.gravity[2]);
    p.rest_threshold = 2.0f * g * p.h;
}

// A contact between bodies a and b, refreshed every substep. The normal n
// points from b towards a; ra / rb are the touching points in each body's
// local frame, so the penetration can be re-measured as the bodies move.
struct Contact {
    int a, b;
    float n[3];
    float ra[3], rb[3];
    float lambda_n;              // accumulated normal correction
    float static_friction[3];    // accumulated static friction correction on a
    float friction_impulse[3];   // accumulated dynamic friction impulse on a
    float vn_pre;              // normal relative velocity when detected
    float mu, e;               // combined friction and restitution
};

// Raw buffer pointers, valid on the side the routines run on.
struct Buffers {
    // Per-body state, [nenv * nbody].
    float* pos[3];
    float* vel[3];
    float* quat[4];  // w, x, y, z
    float* angvel[3];
    uint8_t* enabled;

    // Per-body model, [nbody], shared by all envs.
    const int* shape;
    const float* size;         // [nbody * 3], shape dimensions
    const float* radius;       // bounding-sphere radius
    const float* inv_mass;     // 0 for static bodies
    const float* inv_inertia;  // [nbody * 3], body-frame diagonal
    const float* restitution;
    const float* friction;

    // Particle solver scratch.
    float* dpos[3];
    float* dvel[3];
    int* key;                 // [nenv * nbody] grid key of each body
    const int* sorted_key;    // keys after a stable sort
    const int* sorted_index;  // body index of each sorted slot
    int* cell_start;          // [nenv * ncell] first sorted slot, -1 if empty
    int* cell_end;            // one past the last sorted slot

    // Rigid solver scratch.
    float* prev_pos[3];
    float* prev_quat[4];
    Contact* contacts;  // [nenv * max_contacts]
    int* ncontact;      // [nenv] contacts found (may exceed max_contacts)
};

}  // namespace detail
}  // namespace phys
