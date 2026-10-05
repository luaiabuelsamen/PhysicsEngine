#pragma once

// Elastic contact patch for tactile skins.
//
// Models a sensor's gel (or a fingertip's skin) as a linear elastic
// half-space sampled on a grid of square cells, and computes the pressure
// and shear traction on every cell as a rigid indenter presses and slides
// on it. Unlike rigid contact with Coulomb friction, or per-point spring
// ("Winkler" / hydroelastic) models, cells are coupled through the elastic
// half-space, so the model reproduces Hertz contact (F ~ d^1.5 for a
// sphere) and Cattaneo-Mindlin partial slip (shear builds up over a finite
// sliding distance, starting at the edge of the contact). Measured
// GelSight Mini and DIGIT data follow these laws rather than the Winkler
// ones (tools/sparsh_contact_laws.py).
//
//   normal:      Boussinesq kernel  w_i = sum_j p_j A / (pi E* r_ij)
//   tangential:  Cerruti kernel, angle-averaged
//                u_i = sum_j q_j A (2 - nu)(1 + nu) / (2 pi E r_ij)
//
// Pressure is found with projected Gauss-Seidel (p >= 0, no adhesion).
// Shear is path dependent: every cell keeps a slip state, and the tangential
// solve (Kalker's method) sticks cells whose traction is inside the friction
// cone and slides the rest at |q| = mu p. Normal and tangential problems are
// solved uncoupled, which is exact for an incompressible gel (nu = 0.5).
//
// Frames: the patch lies in the x-y plane with its outward normal along +z;
// the indenter approaches from +z. Lengths in metres, forces in newtons.

#include <vector>

#include "phys/phys.h"

namespace phys {

struct ElasticPatchDesc {
    // HalfSpace couples all cells through the elastic half-space. Winkler
    // ("hydroelastic" / brush) treats every cell as an independent spring;
    // it is kept as a baseline and as a cheap alternative.
    enum Model { HalfSpace, Winkler };
    Model model = HalfSpace;
    float width = 0.02f;           // extent along x
    float height = 0.02f;          // extent along y
    float cell = 2e-4f;            // grid spacing
    float dome_radius = 0.0f;      // > 0: convex surface z = -(x^2 + y^2) / (2 R)
    float youngs_modulus = 1e6f;   // Pa
    float poisson = 0.5f;
    float friction = 0.5f;
    // Multiplies the tangential compliance of the half-space. 1 is a
    // homogeneous elastic half-space; real sensors are often more compliant
    // tangentially (thin coated gels, compliant mounts), which spreads the
    // stick-to-slip transition over a longer slide.
    float tangential_compliance_scale = 1.0f;
    // Winkler model: pressure per unit depth (Pa/m) and shear / normal
    // stiffness ratio.
    float winkler_stiffness = 1e9f;
    float winkler_shear_ratio = 0.3f;
    // Tangential stiffness (N/m) of whatever holds the indenter (robot,
    // mount, force sensor), in series with the contact: the contact moves
    // by tip - shear / stiffness. 0 means rigid.
    float series_tangential_stiffness = 0.0f;
    int max_iterations = 400;      // per solve
    float tolerance = 1e-9f;       // displacement tolerance (m)
};

struct Indenter {
    enum Kind { Sphere, FlatPunch, Cone };
    Kind kind = Sphere;
    float radius = 5e-3f;          // sphere radius, punch radius
    float half_angle = 1.0f;       // cone half angle (rad)
};

class ElasticPatch {
public:
    explicit ElasticPatch(const ElasticPatchDesc& desc);

    // Forget all contact history.
    void reset();

    // Advance to a new indenter position: `tip` is the indenter's lowest
    // point (its axis is along z). Tangential motion of the tip since the
    // previous call drives the shear solve.
    void step(const Indenter& indenter, Vec3 tip);

    // Force exerted by the indenter on the patch: (shear x, shear y, normal),
    // normal positive in compression.
    Vec3 force() const { return force_; }

    int cells() const { return n_; }
    float cell_x(int i) const { return x_[i]; }
    float cell_y(int i) const { return y_[i]; }
    float pressure(int i) const { return p_[i] / area_; }   // Pa
    Vec3 traction(int i) const;  // (qx, qy, p) in Pa
    bool sticking(int i) const { return stick_[i] != 0; }
    int contact_cells() const;

private:
    float kernel(int i, int j, bool tangential) const;
    void solve_normal(const std::vector<float>& interference);
    void solve_tangential(float ux, float uy);

    ElasticPatchDesc d_;
    bool coupled_;  // cells interact (half-space)
    int nx_, ny_, n_;
    float area_, e_star_, normal_self_, tangential_self_, normal_coef_, tangential_coef_;
    std::vector<float> x_, y_, surface_;
    std::vector<float> p_;          // normal force per cell (N)
    std::vector<float> qx_, qy_;    // tangential force per cell (N)
    std::vector<float> sx_, sy_;    // slip state: tip displacement not carried elastically
    std::vector<unsigned char> in_contact_, stick_;
    std::vector<int> active_;       // cells that may touch this step
    float tip_x_ = 0.0f, tip_y_ = 0.0f;
    Vec3 force_{0.0f, 0.0f, 0.0f};
};

}  // namespace phys
