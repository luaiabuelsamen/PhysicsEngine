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

struct Quat {
    float w = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f;
};

enum class Device { CPU, CUDA };

enum class Solver { Rigid, Particle };

enum class Broadphase {  // Particle solver only
    Auto,      // AllPairs for small scenes, Grid otherwise
    AllPairs,  // test every pair of bodies within an env
    Grid,      // uniform spatial hash, keyed per env
};

// Shapes are centred on the body origin. Capsules run along the local y axis;
// planes pass through the body origin and face along local +y. Bodies with
// Shape::None have no collision geometry and need an explicit inertia.
enum class Shape { Sphere = 0, Capsule = 1, Box = 2, Plane = 3, None = 4 };

struct BodyDesc {
    Shape shape = Shape::Sphere;
    // Sphere: {radius}. Capsule: {radius, half_length}. Box: half extents.
    Vec3 size{0.5f, 0.0f, 0.0f};
    float mass = 1.0f;  // <= 0 makes the body static; planes must be static
    float restitution = 0.5f;
    float friction = 0.5f;  // Rigid solver only
    // Principal moments of inertia in the body frame. Zero means "compute
    // from the shape as a solid of uniform density".
    Vec3 inertia{0.0f, 0.0f, 0.0f};
    // Two bodies collide only if each one's group shares a bit with the
    // other's mask (as MuJoCo's contype / conaffinity).
    uint32_t collision_group = 1;
    uint32_t collision_mask = 0xffffffffu;

    static BodyDesc sphere(float radius, float mass);
    static BodyDesc capsule(float radius, float half_length, float mass);
    static BodyDesc box(Vec3 half_extents, float mass);
    static BodyDesc plane();
    static BodyDesc none(float mass, Vec3 inertia);
};

enum class JointType { Hinge = 0, Slider = 1, Ball = 2, Fixed = 3 };

enum class Actuator {
    None = 0,
    Torque = 1,    // control is a torque (hinge) or force (slider)
    Position = 2,  // control is a target angle / offset; PD with gains kp, kd
    Velocity = 3,  // control is a target angular / linear velocity
};

// A joint connects a child body to a parent body (or to the world, parent =
// -1). It is defined by a frame - an anchor point and an orientation - in
// each body's local frame (in world coordinates for the world). The frame's
// x axis is the hinge axis or slider direction. At joint position 0 the two
// frames coincide. Rigid solver only.
struct JointDesc {
    JointType type = JointType::Hinge;
    int parent = -1;
    int child = 0;
    Vec3 parent_anchor{0.0f, 0.0f, 0.0f};
    Quat parent_frame;
    Vec3 child_anchor{0.0f, 0.0f, 0.0f};
    Quat child_frame;

    // Hinge / slider only.
    bool limited = false;
    float lower = 0.0f, upper = 0.0f;  // radians or length
    float damping = 0.0f;              // passive, torque (force) per unit velocity
    Actuator actuator = Actuator::None;
    float kp = 0.0f, kd = 0.0f;        // Position actuator gains
    float max_force = 0.0f;            // actuator force / torque limit; 0: none

    // Joints whose two frames share one orientation, with `axis` (a hinge or
    // slider direction, ignored for ball / fixed) given in the parent frame.
    // Use when parent and child have the same orientation at joint position 0.
    static JointDesc hinge(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor,
                           Vec3 axis);
    static JointDesc slider(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor,
                            Vec3 axis);
    static JointDesc ball(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor);
    static JointDesc fixed(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor);
};

// A tactile sensor: a rectangular pad of elastic gel on the surface of a
// body, sampled on an nx x ny grid (Rigid solver only). The rigid solver
// decides how hard bodies press on the pad; after every step the sensor
// distributes that force over its cells as an elastic gel would - Hertz-like
// pressure under curved objects, partial slip under shear - and reports, per
// cell: pressure, shear traction, the gel's normal deflection (the depth map
// a vision-based sensor sees), its tangential displacement (what markers
// show), and whether the cell sticks. See src/phys/tactile_sensor.h.
//
// The pad lies in the x-y plane of its frame with its outward normal along
// +z; place it on the face of the body's collision shape that touches
// objects.
struct TactileSensorDesc {
    int body = 0;
    Vec3 origin{0.0f, 0.0f, 0.0f};  // pad centre in the body frame
    Quat frame;                     // pad frame in the body frame
    float width = 0.02f;            // along the pad's x axis
    float height = 0.02f;           // along y
    int nx = 16, ny = 16;
    float youngs_modulus = 3e5f;    // Pa (DIGIT's Solaris gel: roughly 0.3 MPa)
    float poisson = 0.5f;
    float dome_radius = 0.0f;       // > 0: the gel bulges out, surface z = -r^2 / 2R
    int max_iterations = 100;       // per contact solve
    float tolerance = 1e-4f;        // relative change of the cell forces
};

struct ModelDesc {
    std::vector<BodyDesc> bodies;
    std::vector<JointDesc> joints;  // Rigid solver only
    bool collide_jointed_bodies = false;  // contacts between a joint's two bodies
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
    std::vector<TactileSensorDesc> tactile_sensors;

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
    float* ctrl;      // [nenv * njoint] actuator inputs
    float* joint_q;   // [nenv * njoint] joint positions, after the last step
    float* joint_qd;  // [nenv * njoint] joint velocities, after the last step
    // Tactile sensors, after the last step: per env, every sensor's block of
    // [kTactileChannels][ny][nx] values (sensors in order), and per env and
    // sensor its total (shear x, shear y, normal) force in the pad frame.
    float* tactile;
    float* tactile_force;
    Device device;
    int nenv, nbody, njoint;
    int nsensor, ntactile;  // sensors; tactile cells per env
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
    int njoint() const { return njoint_; }
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

    // Actuator inputs, [nenv * njoint] (entries for joints without an
    // actuator are ignored). They persist until changed.
    void set_controls(const std::vector<float>& ctrl);
    // Joint positions (hinge angle, slider offset) and velocities after the
    // last step, [nenv * njoint]. Ball and fixed joints report 0.
    void get_joint_state(std::vector<float>& q, std::vector<float>& qd);

    // Tactile sensors. Readings are refreshed at the end of every step() call.
    int nsensor() const { return (int)sensors_.size(); }
    const TactileSensorDesc& sensor(int i) const { return sensors_.at(i); }
    static constexpr int kTactileChannels = 7;  // pressure, shear x/y, deflection, displacement x/y, stick
    // [nenv * ntactile * kTactileChannels] cell readings and
    // [nenv * nsensor * 3] total forces, laid out as in StateView.
    void get_tactile(std::vector<float>& cells, std::vector<float>& forces);

    // Rigid solver: contacts found per env in the last substep. A count above
    // max_contacts_per_env() means contacts were dropped.
    void get_contact_counts(std::vector<int>& counts);

    StateView state();

private:
    int nenv_, nbody_, njoint_;
    Device device_;
    Solver solver_;
    Broadphase broadphase_;
    int max_contacts_;
    int ntactile_ = 0;
    std::vector<TactileSensorDesc> sensors_;
    std::unique_ptr<detail::Backend> backend_;
};

}  // namespace phys
