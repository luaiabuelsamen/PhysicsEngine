#include "phys/phys.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

#include "phys/backend.h"
#include "phys/tactile_sensor.h"

namespace phys {

namespace {

static_assert(World::kTactileChannels == detail::kTactileChannels, "tactile channel count");

// Scenes up to this size default to the all-pairs broadphase.
constexpr int kAllPairsMaxBodies = 32;
// Cap on grid cells per env; beyond it the cells grow instead.
constexpr long long kMaxCellsPerEnv = 1 << 22;
constexpr float kPi = 3.14159265358979f;

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::invalid_argument("phys::World: " + msg);
}

// Radius of a sphere around the body origin that contains the shape.
float bounding_radius(const BodyDesc& b) {
    switch (b.shape) {
        case Shape::Sphere: return b.size.x;
        case Shape::Capsule: return b.size.x + b.size.y;
        case Shape::Box:
            return std::sqrt(b.size.x * b.size.x + b.size.y * b.size.y + b.size.z * b.size.z);
        case Shape::Plane: return INFINITY;
        case Shape::None: return 0.0f;
    }
    return 0.0f;
}

// Body-frame principal moments of inertia: the explicit ones if given,
// else those of the shape as a uniform solid.
Vec3 inertia(const BodyDesc& b) {
    if (b.inertia.x > 0.0f || b.inertia.y > 0.0f || b.inertia.z > 0.0f) return b.inertia;
    float m = b.mass;
    switch (b.shape) {
        case Shape::Sphere: {
            float i = 0.4f * m * b.size.x * b.size.x;
            return {i, i, i};
        }
        case Shape::Box: {
            float x2 = b.size.x * b.size.x, y2 = b.size.y * b.size.y, z2 = b.size.z * b.size.z;
            return {m / 3.0f * (y2 + z2), m / 3.0f * (x2 + z2), m / 3.0f * (x2 + y2)};
        }
        case Shape::Capsule: {
            // Cylinder of length 2h plus two hemispherical caps, split by volume.
            float r = b.size.x, h = b.size.y;
            float v_cyl = kPi * r * r * 2.0f * h;
            float v_caps = 4.0f / 3.0f * kPi * r * r * r;
            float m_cyl = m * v_cyl / (v_cyl + v_caps);
            float m_caps = m - m_cyl;
            float axial = m_cyl * r * r / 2.0f + m_caps * 0.4f * r * r;
            float lateral = m_cyl * (r * r / 4.0f + h * h / 3.0f) +
                            m_caps * (0.4f * r * r + h * h + 0.75f * h * r);
            return {lateral, axial, lateral};
        }
        case Shape::Plane:
        case Shape::None: return {0.0f, 0.0f, 0.0f};
    }
    return {0.0f, 0.0f, 0.0f};
}

// Rotation taking the x axis onto unit vector `axis`.
Quat frame_with_x_axis(Vec3 axis) {
    float n = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    require(n > 0.0f, "joint axis must be non-zero");
    float x = axis.x / n, y = axis.y / n, z = axis.z / n;
    if (x < -0.999999f) return {0.0f, 0.0f, 1.0f, 0.0f};  // half turn about y
    // q = normalize(1 + x.axis, x cross axis) with x = (1, 0, 0)
    float w = 1.0f + x, qy = -z, qz = y;
    float len = std::sqrt(w * w + qy * qy + qz * qz);
    return {w / len, 0.0f, qy / len, qz / len};
}

JointDesc make_joint(JointType type, int parent, int child, Vec3 parent_anchor,
                     Vec3 child_anchor, Vec3 axis) {
    JointDesc j;
    j.type = type;
    j.parent = parent;
    j.child = child;
    j.parent_anchor = parent_anchor;
    j.child_anchor = child_anchor;
    j.parent_frame = j.child_frame = frame_with_x_axis(axis);
    return j;
}

detail::JointModel to_model(const JointDesc& j) {
    detail::JointModel m{};
    m.type = (int)j.type;
    m.parent = j.parent;
    m.child = j.child;
    const Vec3 anchors[2] = {j.parent_anchor, j.child_anchor};
    const Quat frames[2] = {j.parent_frame, j.child_frame};
    float* anchor_out[2] = {m.parent_anchor, m.child_anchor};
    float* frame_out[2] = {m.parent_frame, m.child_frame};
    for (int s = 0; s < 2; s++) {
        anchor_out[s][0] = anchors[s].x;
        anchor_out[s][1] = anchors[s].y;
        anchor_out[s][2] = anchors[s].z;
        const Quat& q = frames[s];
        float n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        require(n > 0.0f, "joint frame quaternion must be non-zero");
        frame_out[s][0] = q.w / n;
        frame_out[s][1] = q.x / n;
        frame_out[s][2] = q.y / n;
        frame_out[s][3] = q.z / n;
    }
    m.limited = j.limited ? 1 : 0;
    m.lower = j.lower;
    m.upper = j.upper;
    m.damping = j.damping;
    m.actuator = (int)j.actuator;
    m.kp = j.kp;
    m.kd = j.kd;
    m.max_force = j.max_force;
    return m;
}

// Half-space influence coefficients of a sensor's grid: normal deflection
// at a cell per unit force on a cell (dx, dy) cells away. The cell's own
// coefficient is the centre deflection under a uniformly loaded rectangle;
// the others treat the force as a point load.
void add_tactile_sensor(const TactileSensorDesc& t, int nenv, detail::ModelArrays& model) {
    require(t.width > 0.0f && t.height > 0.0f, "tactile sensor sizes must be positive");
    require(t.nx > 0 && t.ny > 0 && (long long)t.nx * t.ny <= 4096,
            "tactile sensors need 1 to 4096 cells (64 x 64)");
    require(t.youngs_modulus > 0.0f && t.poisson >= 0.0f && t.poisson < 0.5001f,
            "tactile sensors need E > 0 and 0 <= poisson <= 0.5");
    require(t.max_iterations > 0 && t.tolerance > 0.0f, "tactile solver settings must be positive");
    require(t.dome_radius >= 0.0f, "tactile dome radius must be non-negative");

    detail::TactileModel m{};
    m.body = t.body;
    m.origin[0] = t.origin.x;
    m.origin[1] = t.origin.y;
    m.origin[2] = t.origin.z;
    float qn = std::sqrt(t.frame.w * t.frame.w + t.frame.x * t.frame.x + t.frame.y * t.frame.y +
                         t.frame.z * t.frame.z);
    require(qn > 0.0f, "tactile sensor frame quaternion must be non-zero");
    m.frame[0] = t.frame.w / qn;
    m.frame[1] = t.frame.x / qn;
    m.frame[2] = t.frame.y / qn;
    m.frame[3] = t.frame.z / qn;
    m.half_w = 0.5f * t.width;
    m.half_h = 0.5f * t.height;
    m.nx = t.nx;
    m.ny = t.ny;
    m.cell_x = t.width / (float)t.nx;
    m.cell_y = t.height / (float)t.ny;
    m.dome_radius = t.dome_radius;
    float nu = std::fmin(t.poisson, 0.5f);
    m.tangential_ratio = (2.0f - nu) / (2.0f * (1.0f - nu));
    m.max_iterations = t.max_iterations;
    m.tolerance = t.tolerance;

    m.cell_offset = 0;
    for (const detail::TactileModel& other : model.sensors) m.cell_offset += other.nx * other.ny;
    m.kernel_offset = (int)model.tactile_kernel.size();
    m.scratch_offset = model.tactile_scratch;
    m.list_offset = model.tactile_list;
    long long cells = (long long)t.nx * t.ny;
    model.tactile_scratch += 7 * cells * nenv;  // detail::kTactileArrays
    model.tactile_list += cells * nenv;

    double e_star = t.youngs_modulus / (1.0 - (double)nu * nu);
    double a = 0.5 * m.cell_x, b = 0.5 * m.cell_y, d = std::sqrt(a * a + b * b);
    double self = 4.0 * (a * std::log((b + d) / a) + b * std::log((a + d) / b)) /
                  (kPi * e_star * m.cell_x * m.cell_y);
    for (int dy = 0; dy < t.ny; dy++) {
        for (int dx = 0; dx < t.nx; dx++) {
            double r = std::hypot(dx * (double)m.cell_x, dy * (double)m.cell_y);
            model.tactile_kernel.push_back((float)(dx == 0 && dy == 0 ? self : 1.0 / (kPi * e_star * r)));
        }
    }
    model.sensors.push_back(m);
}

}  // namespace

JointDesc JointDesc::hinge(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor,
                           Vec3 axis) {
    return make_joint(JointType::Hinge, parent, child, parent_anchor, child_anchor, axis);
}

JointDesc JointDesc::slider(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor,
                            Vec3 axis) {
    return make_joint(JointType::Slider, parent, child, parent_anchor, child_anchor, axis);
}

JointDesc JointDesc::ball(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor) {
    return make_joint(JointType::Ball, parent, child, parent_anchor, child_anchor, {1, 0, 0});
}

JointDesc JointDesc::fixed(int parent, int child, Vec3 parent_anchor, Vec3 child_anchor) {
    return make_joint(JointType::Fixed, parent, child, parent_anchor, child_anchor, {1, 0, 0});
}

BodyDesc BodyDesc::sphere(float radius, float mass) {
    BodyDesc b;
    b.shape = Shape::Sphere;
    b.size = {radius, 0.0f, 0.0f};
    b.mass = mass;
    return b;
}

BodyDesc BodyDesc::capsule(float radius, float half_length, float mass) {
    BodyDesc b;
    b.shape = Shape::Capsule;
    b.size = {radius, half_length, 0.0f};
    b.mass = mass;
    return b;
}

BodyDesc BodyDesc::box(Vec3 half_extents, float mass) {
    BodyDesc b;
    b.shape = Shape::Box;
    b.size = half_extents;
    b.mass = mass;
    return b;
}

BodyDesc BodyDesc::none(float mass, Vec3 inertia) {
    BodyDesc b;
    b.shape = Shape::None;
    b.size = {0.0f, 0.0f, 0.0f};
    b.mass = mass;
    b.inertia = inertia;
    return b;
}

BodyDesc BodyDesc::plane() {
    BodyDesc b;
    b.shape = Shape::Plane;
    b.size = {0.0f, 0.0f, 0.0f};
    b.mass = 0.0f;
    return b;
}

void HostState::resize(int n) {
    for (auto* v : {&px, &py, &pz, &vx, &vy, &vz, &qx, &qy, &qz, &wx, &wy, &wz})
        v->assign(n, 0.0f);
    qw.assign(n, 1.0f);
    enabled.assign(n, 1);
}

World::World(const ModelDesc& desc, int nenv, Device device)
    : nenv_(nenv), nbody_((int)desc.bodies.size()), njoint_((int)desc.joints.size()),
      device_(device), solver_(desc.solver),
      broadphase_(Broadphase::AllPairs), max_contacts_(0) {
    require(nenv_ > 0, "nenv must be positive");
    require(nbody_ > 0, "the model has no bodies");
    require((long long)nenv_ * nbody_ <= INT_MAX, "too many bodies in total");

    detail::ModelArrays model;
    for (const BodyDesc& b : desc.bodies) {
        bool dynamic = b.mass > 0.0f;
        switch (b.shape) {
            case Shape::Sphere: require(b.size.x > 0.0f, "sphere radius must be positive"); break;
            case Shape::Capsule:
                require(b.size.x > 0.0f && b.size.y >= 0.0f,
                        "capsule needs a positive radius and non-negative half length");
                break;
            case Shape::Box:
                require(b.size.x > 0.0f && b.size.y > 0.0f && b.size.z > 0.0f,
                        "box half extents must be positive");
                break;
            case Shape::Plane: require(!dynamic, "planes must be static (mass <= 0)"); break;
            case Shape::None: break;
        }
        if (solver_ == Solver::Particle)
            require(b.shape == Shape::Sphere, "the particle solver only supports spheres");
        Vec3 I = inertia(b);
        require(!dynamic || (I.x > 0.0f && I.y > 0.0f && I.z > 0.0f),
                "dynamic bodies need positive inertia (give one for Shape::None)");
        model.shape.push_back((int)b.shape);
        model.size.insert(model.size.end(), {b.size.x, b.size.y, b.size.z});
        model.radius.push_back(bounding_radius(b));
        model.inv_mass.push_back(dynamic ? 1.0f / b.mass : 0.0f);
        model.inv_inertia.insert(model.inv_inertia.end(),
                                 {dynamic ? 1.0f / I.x : 0.0f, dynamic ? 1.0f / I.y : 0.0f,
                                  dynamic ? 1.0f / I.z : 0.0f});
        model.restitution.push_back(b.restitution);
        model.friction.push_back(b.friction);
    }

    detail::Params p{};
    p.nenv = nenv_;
    p.nbody = nbody_;
    p.rigid = solver_ == Solver::Rigid;
    p.gravity[0] = desc.gravity.x;
    p.gravity[1] = desc.gravity.y;
    p.gravity[2] = desc.gravity.z;

    if (p.rigid) {
        // Pairs allowed to collide: group / mask bits, minus jointed pairs.
        model.may_collide.assign((size_t)nbody_ * nbody_, 0);
        for (int i = 0; i < nbody_; i++) {
            for (int j = 0; j < nbody_; j++) {
                const BodyDesc &bi = desc.bodies[i], &bj = desc.bodies[j];
                bool allowed = i != j && (bi.collision_group & bj.collision_mask) &&
                               (bj.collision_group & bi.collision_mask);
                model.may_collide[(size_t)i * nbody_ + j] = allowed ? 1 : 0;
            }
        }
        for (const JointDesc& j : desc.joints) {
            require(j.child >= 0 && j.child < nbody_, "joint child out of range");
            require(j.parent >= -1 && j.parent < nbody_ && j.parent != j.child,
                    "joint parent must be another body or -1 (the world)");
            bool axial = j.type == JointType::Hinge || j.type == JointType::Slider;
            require(axial || (!j.limited && j.actuator == Actuator::None && j.damping == 0.0f),
                    "limits, damping and actuators need a hinge or slider joint");
            require(!j.limited || j.lower <= j.upper, "joint limits need lower <= upper");
            require(j.kp >= 0.0f && j.kd >= 0.0f && j.damping >= 0.0f && j.max_force >= 0.0f,
                    "joint gains, damping and force limits must be non-negative");
            model.joints.push_back(to_model(j));
            p.has_torque_actuators |= j.actuator == Actuator::Torque;
            if (j.parent >= 0 && !desc.collide_jointed_bodies) {
                model.may_collide[(size_t)j.parent * nbody_ + j.child] = 0;
                model.may_collide[(size_t)j.child * nbody_ + j.parent] = 0;
            }
        }
        p.njoint = njoint_;

        require(desc.substeps > 0, "substeps must be positive");
        p.substeps = desc.substeps;
        require(desc.position_iterations > 0 && desc.velocity_iterations > 0,
                "solver iteration counts must be positive");
        p.position_iterations = desc.position_iterations;
        p.velocity_iterations = desc.velocity_iterations;
        max_contacts_ = desc.max_contacts_per_env > 0 ? desc.max_contacts_per_env
                                                      : std::max(16, 8 * nbody_);
        require((long long)max_contacts_ * nenv_ <= INT_MAX, "contact buffer too large");
        p.max_contacts = max_contacts_;

        for (const TactileSensorDesc& t : desc.tactile_sensors) {
            require(t.body >= 0 && t.body < nbody_, "tactile sensor body out of range");
            add_tactile_sensor(t, nenv_, model);
        }
        sensors_ = desc.tactile_sensors;
        p.nsensor = (int)sensors_.size();
        p.ntactile = ntactile_ = 0;
        for (const detail::TactileModel& t : model.sensors) ntactile_ += t.nx * t.ny;
        p.ntactile = ntactile_;
        p.max_sensor_cells = 0;
        for (const detail::TactileModel& t : model.sensors)
            p.max_sensor_cells = std::max(p.max_sensor_cells, t.nx * t.ny);
        p.tactile_threads = detail::tactile_threads_for(p.max_sensor_cells);
    } else {
        require(njoint_ == 0, "the particle solver does not support joints");
        require(desc.tactile_sensors.empty(), "tactile sensors need the rigid solver");
        p.substeps = 1;
        const float lo[3] = {desc.bounds_lo.x, desc.bounds_lo.y, desc.bounds_lo.z};
        const float hi[3] = {desc.bounds_hi.x, desc.bounds_hi.y, desc.bounds_hi.z};
        for (int a = 0; a < 3; a++) {
            require(hi[a] > lo[a], "bounds_hi must exceed bounds_lo on every axis");
            p.lo[a] = lo[a];
            p.hi[a] = hi[a];
        }
        p.wall_restitution = desc.wall_restitution;
        p.contact_correction = desc.contact_correction;

        broadphase_ = desc.broadphase;
        if (broadphase_ == Broadphase::Auto)
            broadphase_ = nbody_ <= kAllPairsMaxBodies ? Broadphase::AllPairs : Broadphase::Grid;
        p.grid = broadphase_ == Broadphase::Grid;

        if (p.grid) {
            // A cell must be at least one body diameter wide so that every
            // contact is found among the 27 surrounding cells.
            float cell = 2.0f * *std::max_element(model.radius.begin(), model.radius.end());
            long long ncell;
            while (true) {
                p.inv_cell = 1.0f / cell;
                ncell = 1;
                for (int a = 0; a < 3; a++) {
                    p.dim[a] = std::max(1, (int)std::ceil((hi[a] - lo[a]) * p.inv_cell));
                    ncell *= p.dim[a];
                }
                if (ncell <= kMaxCellsPerEnv) break;
                cell *= 1.25f;
            }
            require(ncell * nenv_ < INT_MAX,
                    "grid has too many cells across all envs; use Broadphase::AllPairs "
                    "or smaller bounds");
            p.ncell = (int)ncell;
            p.sentinel_key = p.ncell * nenv_;
        }
    }

    backend_ = device_ == Device::CUDA ? detail::make_cuda_backend(p, model)
                                       : detail::make_cpu_backend(p, model);
}

World::~World() = default;
World::World(World&&) noexcept = default;
World& World::operator=(World&&) noexcept = default;

void World::set_state(const HostState& s) {
    size_t n = (size_t)nenv_ * nbody_;
    bool ok = s.enabled.size() == n;
    for (const auto* v : {&s.px, &s.py, &s.pz, &s.vx, &s.vy, &s.vz, &s.qw, &s.qx, &s.qy, &s.qz,
                          &s.wx, &s.wy, &s.wz})
        ok = ok && v->size() == n;
    require(ok, "state arrays must have nenv * nbody entries");
    backend_->upload(s);
}

void World::get_state(HostState& state) {
    state.resize(nenv_ * nbody_);
    backend_->download(state);
}

void World::step(float dt, int nsteps) {
    require(nsteps >= 0, "nsteps must be non-negative");
    require(dt > 0.0f, "dt must be positive");
    backend_->step(dt, nsteps);
}

void World::synchronize() { backend_->synchronize(); }

void World::set_controls(const std::vector<float>& ctrl) {
    require(ctrl.size() == (size_t)nenv_ * njoint_, "controls must have nenv * njoint entries");
    if (njoint_ > 0) backend_->upload_controls(ctrl);
}

void World::get_joint_state(std::vector<float>& q, std::vector<float>& qd) {
    q.assign((size_t)nenv_ * njoint_, 0.0f);
    qd.assign((size_t)nenv_ * njoint_, 0.0f);
    if (njoint_ > 0) backend_->download_joint_state(q, qd);
}

void World::get_contact_counts(std::vector<int>& counts) {
    counts.assign(nenv_, 0);
    if (solver_ == Solver::Rigid) backend_->download_contact_counts(counts);
}

void World::get_tactile(std::vector<float>& cells, std::vector<float>& forces) {
    cells.assign((size_t)nenv_ * ntactile_ * kTactileChannels, 0.0f);
    forces.assign((size_t)nenv_ * sensors_.size() * 3, 0.0f);
    if (!sensors_.empty()) backend_->download_tactile(cells, forces);
}

StateView World::state() { return backend_->view(); }

}  // namespace phys
