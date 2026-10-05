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

// Same values as phys::Shape, phys::JointType and phys::Actuator.
enum ShapeType : int { kSphere = 0, kCapsule = 1, kBox = 2, kPlane = 3, kNoShape = 4 };
enum JointKind : int { kHinge = 0, kSlider = 1, kBall = 2, kFixed = 3 };
enum ActuatorKind : int { kNoActuator = 0, kTorque = 1, kPositionDrive = 2, kVelocityDrive = 3 };

// A joint as the solvers see it. Bodies are indices within an env; parent
// -1 is the world, whose "local" frame is the world frame.
struct JointModel {
    int type, parent, child;
    float parent_anchor[3], parent_frame[4];  // frame quaternions are w, x, y, z
    float child_anchor[3], child_frame[4];
    int limited;
    float lower, upper;
    float damping;
    int actuator;
    float kp, kd, max_force;
};

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
    int njoint;
    bool has_torque_actuators;
    int substeps;
    int position_iterations;
    int velocity_iterations;
    float h;               // substep length, dt / substeps
    float rest_threshold;  // below this approach speed, contacts don't bounce
    int max_contacts;      // contact buffer capacity per env

    // Tactile sensors.
    int nsensor;
    int ntactile;          // tactile cells per env, all sensors
    int max_sensor_cells;  // cells of the largest sensor
    bool tactile_shared_work;  // CUDA: tactile working arrays in shared memory
    int tactile_threads;       // threads per tactile solve (see tactile_sensor.h)
};

// Number of output channels of a tactile sensor cell (see tactile_sensor.h).
constexpr int kTactileChannels = 7;
constexpr int kTactileAccum = 5;
constexpr int kTactileMaxTouching = 4;  // bodies considered per pad

// A tactile sensor pad as the solver sees it (see phys::TactileSensorDesc).
// Cells are numbered row-major, cell (ix, iy) = iy * nx + ix, at local
// (-half_w + (ix + 0.5) * cell_x, -half_h + (iy + 0.5) * cell_y).
struct TactileModel {
    int body;
    float origin[3];   // pad centre in the body frame
    float frame[4];    // pad frame in the body frame (w, x, y, z); +z is the outward normal
    float half_w, half_h;
    float cell_x, cell_y;
    int nx, ny;
    float dome_radius;          // > 0: the gel surface bulges out by -r^2 / 2R
    float tangential_ratio;     // tangential / normal compliance of the gel
    int cell_offset;            // first cell of this sensor among an env's tactile cells
    int kernel_offset;          // into the kernel table: [ny * nx] influence coefficients
    long long scratch_offset;   // into the float scratch: 7 * nx * ny per env
    long long list_offset;      // into the int scratch: nx * ny per env
    int max_iterations;
    float tolerance;
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
    float normal_impulse;        // accumulated restitution (normal velocity) impulse on a
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

    // Rigid solver model and inputs.
    const JointModel* joints;   // [njoint]
    const uint8_t* may_collide; // [nbody * nbody] pair filter
    float* ctrl;                // [nenv * njoint]
    float* joint_q;             // [nenv * njoint] outputs
    float* joint_qd;

    // Rigid solver scratch.
    float* joint_lambda;        // [nenv * njoint] accumulated drive correction
    // Motion of each body within the current substep - the prediction plus
    // every correction - kept separately so velocities can be computed as
    // disp / h without subtracting two nearly equal positions in float32.
    float* disp[3];  // linear displacement
    float* drot[3];  // rotation vector
    Contact* contacts;  // [nenv * max_contacts]
    int* ncontact;      // [nenv] contacts found (may exceed max_contacts)

    // Tactile sensors.
    const TactileModel* sensors;  // [nsensor]
    const float* tactile_kernel;  // influence coefficients, see TactileModel
    float* tactile;               // [nenv * ntactile * kTactileChannels] outputs
    float* tactile_force;         // [nenv * nsensor * 3] outputs
    float* tactile_scratch;
    int* tactile_list;
    // Contact force on each pad summed over the substeps of the current step:
    // [nenv * nsensor * kTactileAccum] (force x, y, z in world, sum mu lambda,
    // sum lambda) and the bodies touching it, [nenv * nsensor * 4] (-1: none).
    float* tactile_accum;
    int* tactile_touch;
};

}  // namespace detail
}  // namespace phys
