#pragma once

// libphys public API.
//
// A World simulates `nenv` independent copies of one scene. The scene layout
// (bodies, their shapes and masses) is described once by a ModelDesc and
// shared by every env; the per-env state lives in flat arrays indexed by
// `env * nbody + body`. Bodies in different envs never interact.
//
// Two solvers are available:
//   Solver::Rigid     rotating bodies with sphere / capsule / box / plane
//                     shapes, friction and restitution (XPBD with substeps).
//                     The intended solver for RL environments.
//   Solver::Particle  non-rotating frictionless spheres inside axis-aligned
//                     walls; scales to very large single scenes.
//
// Both backends (CPU and CUDA) run the same routines. They are
// deterministic, and with PHYS_STRICT_FP they produce bit-identical
// trajectories, so the CPU backend doubles as a reference for the GPU one.

#include <cstdint>
#include <memory>
#include <vector>

namespace phys {

struct Vec3 {
    float x, y, z;
};

enum class Device { CPU, CUDA };

enum class Solver { Rigid, Particle };

enum class Broadphase {  // Particle solver only
    Auto,      // AllPairs for small scenes, Grid otherwise
    AllPairs,  // test every pair of bodies within an env
    Grid,      // uniform spatial hash, keyed per env
};

// Shapes are centred on the body origin. Capsules run along the local y axis;
// planes pass through the body origin and face along local +y.
enum class Shape { Sphere = 0, Capsule = 1, Box = 2, Plane = 3 };

struct BodyDesc {
    Shape shape = Shape::Sphere;
    // Sphere: {radius}. Capsule: {radius, half_length}. Box: half extents.
    Vec3 size{0.5f, 0.0f, 0.0f};
    float mass = 1.0f;  // <= 0 makes the body static; planes must be static
    float restitution = 0.5f;
    float friction = 0.5f;  // Rigid solver only

    static BodyDesc sphere(float radius, float mass);
    static BodyDesc capsule(float radius, float half_length, float mass);
    static BodyDesc box(Vec3 half_extents, float mass);
    static BodyDesc plane();
};

struct ModelDesc {
    std::vector<BodyDesc> bodies;
    Vec3 gravity{0.0f, -9.81f, 0.0f};
    Solver solver = Solver::Rigid;

    // Rigid solver.
    int substeps = 10;
    // Gauss-Seidel passes over each env's contacts per substep: the position
    // solve (penetration, static friction) and the velocity solve (dynamic
    // friction, restitution).
    int position_iterations = 4;
    int velocity_iterations = 4;
    int max_contacts_per_env = 0;  // 0: 8 per body, at least 16

    // Particle solver: axis-aligned walls that bodies bounce off.
    Vec3 bounds_lo{0.0f, 0.0f, 0.0f};
    Vec3 bounds_hi{1.0f, 1.0f, 1.0f};
    float wall_restitution = 0.9f;
    // Fraction of each overlap removed per step by positional correction.
    float contact_correction = 0.5f;
    Broadphase broadphase = Broadphase::Auto;
};

// Host copy of the state of every env. Arrays have nenv * nbody entries.
struct HostState {
    std::vector<float> px, py, pz;      // position
    std::vector<float> vx, vy, vz;      // linear velocity
    std::vector<float> qw, qx, qy, qz;  // orientation (unit quaternion)
    std::vector<float> wx, wy, wz;      // angular velocity, world frame
    std::vector<uint8_t> enabled;       // disabled bodies neither move nor collide

    HostState() = default;
    explicit HostState(int n) { resize(n); }
    void resize(int n);  // zero motion, identity orientation, enabled
    int size() const { return (int)px.size(); }
};

// Raw pointers to the World's live state buffers, on `device`. They stay valid
// (and unchanged) for the lifetime of the World.
struct StateView {
    float *px, *py, *pz;
    float *vx, *vy, *vz;
    float *qw, *qx, *qy, *qz;
    float *wx, *wy, *wz;
    uint8_t* enabled;
    Device device;
    int nenv, nbody;
};

namespace detail { class Backend; }

class World {
public:
    World(const ModelDesc& desc, int nenv, Device device);
    ~World();
    World(World&&) noexcept;
    World& operator=(World&&) noexcept;
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    int nenv() const { return nenv_; }
    int nbody() const { return nbody_; }
    Device device() const { return device_; }
    Solver solver() const { return solver_; }
    Broadphase broadphase() const { return broadphase_; }  // never Auto
    int max_contacts_per_env() const { return max_contacts_; }

    void set_state(const HostState& state);
    void get_state(HostState& state);

    // Advance every env by `nsteps` steps of length `dt`. On CUDA this is
    // asynchronous; get_state() or synchronize() wait for it.
    void step(float dt, int nsteps = 1);
    void synchronize();

    // Rigid solver: contacts found per env in the last substep. A count above
    // max_contacts_per_env() means contacts were dropped.
    void get_contact_counts(std::vector<int>& counts);

    StateView state();

private:
    int nenv_, nbody_;
    Device device_;
    Solver solver_;
    Broadphase broadphase_;
    int max_contacts_;
    std::unique_ptr<detail::Backend> backend_;
};

}  // namespace phys
