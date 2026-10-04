#include "phys/phys.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

#include "phys/backend.h"

namespace phys {

namespace {

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
    }
    return 0.0f;
}

// Body-frame principal moments of inertia of a solid shape.
Vec3 inertia(const BodyDesc& b) {
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
        case Shape::Plane: return {0.0f, 0.0f, 0.0f};
    }
    return {0.0f, 0.0f, 0.0f};
}

}  // namespace

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
    : nenv_(nenv), nbody_((int)desc.bodies.size()), device_(device), solver_(desc.solver),
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
        }
        if (solver_ == Solver::Particle)
            require(b.shape == Shape::Sphere, "the particle solver only supports spheres");
        model.shape.push_back((int)b.shape);
        model.size.insert(model.size.end(), {b.size.x, b.size.y, b.size.z});
        model.radius.push_back(bounding_radius(b));
        model.inv_mass.push_back(dynamic ? 1.0f / b.mass : 0.0f);
        Vec3 I = inertia(b);
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
    } else {
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

void World::get_contact_counts(std::vector<int>& counts) {
    counts.assign(nenv_, 0);
    if (solver_ == Solver::Rigid) backend_->download_contact_counts(counts);
}

StateView World::state() { return backend_->view(); }

}  // namespace phys
