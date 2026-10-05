#include "phys/tactile.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace phys {

namespace {

constexpr float kPi = 3.14159265358979f;
// Centre deflection of a uniformly loaded square of side c on a half-space,
// per unit force: 4 ln(1 + sqrt 2) / pi / (E* c).
constexpr float kSquareSelf = 1.12220f;

}  // namespace

ElasticPatch::ElasticPatch(const ElasticPatchDesc& desc) : d_(desc) {
    if (d_.cell <= 0.0f || d_.width <= 0.0f || d_.height <= 0.0f)
        throw std::invalid_argument("ElasticPatch: sizes must be positive");
    if (d_.youngs_modulus <= 0.0f || d_.poisson < 0.0f || d_.poisson > 0.5f)
        throw std::invalid_argument("ElasticPatch: need E > 0 and 0 <= poisson <= 0.5");
    nx_ = std::max(1, (int)std::lround(d_.width / d_.cell));
    ny_ = std::max(1, (int)std::lround(d_.height / d_.cell));
    n_ = nx_ * ny_;
    area_ = d_.cell * d_.cell;
    float nu = d_.poisson;
    e_star_ = d_.youngs_modulus / (1.0f - nu * nu);
    normal_coef_ = 1.0f / (kPi * e_star_);
    // Angle-averaged Cerruti kernel, (2 - nu)(1 + nu) / (2 pi E r), in terms of E*.
    if (d_.tangential_compliance_scale <= 0.0f || d_.series_tangential_stiffness < 0.0f)
        throw std::invalid_argument("ElasticPatch: compliance scales must be positive");
    float tangential_ratio = (2.0f - nu) / (2.0f * (1.0f - nu)) * d_.tangential_compliance_scale;
    tangential_coef_ = normal_coef_ * tangential_ratio;
    normal_self_ = kSquareSelf / (e_star_ * d_.cell);
    tangential_self_ = normal_self_ * tangential_ratio;
    coupled_ = d_.model == ElasticPatchDesc::HalfSpace;
    if (!coupled_) {
        if (d_.winkler_stiffness <= 0.0f || d_.winkler_shear_ratio <= 0.0f)
            throw std::invalid_argument("ElasticPatch: Winkler stiffnesses must be positive");
        normal_self_ = 1.0f / (d_.winkler_stiffness * area_);
        tangential_self_ = normal_self_ / d_.winkler_shear_ratio;
    }

    x_.resize(n_);
    y_.resize(n_);
    surface_.resize(n_);
    for (int j = 0; j < ny_; j++) {
        for (int i = 0; i < nx_; i++) {
            int k = j * nx_ + i;
            x_[k] = (i + 0.5f) * d_.cell - 0.5f * nx_ * d_.cell;
            y_[k] = (j + 0.5f) * d_.cell - 0.5f * ny_ * d_.cell;
            float r2 = x_[k] * x_[k] + y_[k] * y_[k];
            surface_[k] = d_.dome_radius > 0.0f ? -r2 / (2.0f * d_.dome_radius) : 0.0f;
        }
    }
    reset();
}

void ElasticPatch::reset() {
    p_.assign(n_, 0.0f);
    qx_.assign(n_, 0.0f);
    qy_.assign(n_, 0.0f);
    sx_.assign(n_, 0.0f);
    sy_.assign(n_, 0.0f);
    in_contact_.assign(n_, 0);
    stick_.assign(n_, 0);
    active_.clear();
    force_ = {0.0f, 0.0f, 0.0f};
}

float ElasticPatch::kernel(int i, int j, bool tangential) const {
    if (i == j) return tangential ? tangential_self_ : normal_self_;
    if (!coupled_) return 0.0f;
    float dx = x_[i] - x_[j], dy = y_[i] - y_[j];
    float r = std::sqrt(dx * dx + dy * dy);
    return (tangential ? tangential_coef_ : normal_coef_) / r;
}

Vec3 ElasticPatch::traction(int i) const {
    return {qx_[i] / area_, qy_[i] / area_, p_[i] / area_};
}

int ElasticPatch::contact_cells() const {
    int c = 0;
    for (int i = 0; i < n_; i++) c += p_[i] > 0.0f;
    return c;
}

void ElasticPatch::step(const Indenter& ind, Vec3 tip) {
    // Overlap between the undeformed surface and the indenter; contact can
    // only occur where it is positive.
    std::vector<float> interference(n_, -1.0f);
    std::vector<int> active;
    for (int k = 0; k < n_; k++) {
        float dx = x_[k] - tip.x, dy = y_[k] - tip.y;
        float rho2 = dx * dx + dy * dy;
        float z;  // height of the indenter's surface above this cell
        if (ind.kind == Indenter::Sphere) {
            if (rho2 >= ind.radius * ind.radius) continue;
            z = tip.z + ind.radius - std::sqrt(ind.radius * ind.radius - rho2);
        } else if (ind.kind == Indenter::FlatPunch) {
            if (rho2 > ind.radius * ind.radius) continue;
            z = tip.z;
        } else {
            z = tip.z + std::sqrt(rho2) / std::tan(ind.half_angle);
        }
        float h = surface_[k] - z;
        if (h > 0.0f) {
            interference[k] = h;
            active.push_back(k);
        }
    }
    // Cells that left the overlap region carry no load.
    for (int k : active_) {
        if (interference[k] <= 0.0f) {
            p_[k] = qx_[k] = qy_[k] = 0.0f;
            in_contact_[k] = stick_[k] = 0;
        }
    }
    active_ = active;

    solve_normal(interference);
    solve_tangential(tip.x, tip.y);

    force_ = {0.0f, 0.0f, 0.0f};
    for (int k : active_) {
        force_.x += qx_[k];
        force_.y += qy_[k];
        force_.z += p_[k];
    }
    tip_x_ = tip.x;
    tip_y_ = tip.y;
}

// Projected Gauss-Seidel on  w = C p,  p >= 0,  (w - h) p = 0,  w >= h,
// warm-started from the previous step's pressures.
void ElasticPatch::solve_normal(const std::vector<float>& h) {
    int na = (int)active_.size();
    if (na == 0) return;
    std::vector<float> w(na, 0.0f);
    for (int a = 0; a < na; a++) {
        float pa = p_[active_[a]];
        if (pa == 0.0f) continue;
        if (!coupled_) {
            w[a] = normal_self_ * pa;
            continue;
        }
        for (int b = 0; b < na; b++) w[b] += kernel(active_[b], active_[a], false) * pa;
    }
    for (int it = 0; it < d_.max_iterations; it++) {
        float worst = 0.0f;
        for (int a = 0; a < na; a++) {
            int k = active_[a];
            float updated = std::max(0.0f, p_[k] + (h[k] - w[a]) / normal_self_);
            float delta = updated - p_[k];
            if (delta == 0.0f) continue;
            p_[k] = updated;
            if (coupled_) {
                for (int b = 0; b < na; b++) w[b] += kernel(active_[b], k, false) * delta;
            } else {
                w[a] += normal_self_ * delta;
            }
            worst = std::max(worst, std::fabs(delta) * normal_self_);
        }
        if (worst < d_.tolerance) break;
    }
}

// Kalker-style path-dependent tangential solve. A cell sticks if the traction
// needed to carry the tip's displacement elastically, minus the slip it has
// already accumulated, lies inside the friction cone; otherwise it slides at
// |q| = mu p and its slip state absorbs the excess.
void ElasticPatch::solve_tangential(float tip_x, float tip_y) {
    std::vector<int> contact;
    for (int k : active_) {
        if (p_[k] > 0.0f) {
            contact.push_back(k);
        } else {
            qx_[k] = qy_[k] = 0.0f;
            in_contact_[k] = stick_[k] = 0;
        }
    }
    int nc = (int)contact.size();
    if (nc == 0) return;

    // Elastic tangential displacement of each contact cell from all tractions.
    std::vector<float> ux(nc, 0.0f), uy(nc, 0.0f);
    float total_x = 0.0f, total_y = 0.0f;
    for (int a = 0; a < nc; a++) {
        int k = contact[a];
        total_x += qx_[k];
        total_y += qy_[k];
        if (qx_[k] == 0.0f && qy_[k] == 0.0f) continue;
        if (!coupled_) {
            ux[a] = tangential_self_ * qx_[k];
            uy[a] = tangential_self_ * qy_[k];
            continue;
        }
        for (int b = 0; b < nc; b++) {
            float c = kernel(contact[b], k, true);
            ux[b] += c * qx_[k];
            uy[b] += c * qy_[k];
        }
    }
    // With a compliant holder the contact moves by the tip's displacement
    // less the holder's deflection under the total shear; solved together
    // with the tractions below.
    float series = d_.series_tangential_stiffness > 0.0f ? 1.0f / d_.series_tangential_stiffness : 0.0f;
    float cx = tip_x - series * total_x, cy = tip_y - series * total_y;
    // Cells that just came into contact start unloaded: their slip state
    // absorbs whatever displacement they would otherwise inherit.
    for (int a = 0; a < nc; a++) {
        int k = contact[a];
        if (!in_contact_[k]) {
            float self_x = tangential_self_ * qx_[k], self_y = tangential_self_ * qy_[k];
            sx_[k] = cx - (ux[a] - self_x);
            sy_[k] = cy - (uy[a] - self_y);
            in_contact_[k] = 1;
        }
    }

    for (int it = 0; it < d_.max_iterations; it++) {
        float worst = 0.0f;
        for (int a = 0; a < nc; a++) {
            int k = contact[a];
            float other_x = ux[a] - tangential_self_ * qx_[k];
            float other_y = uy[a] - tangential_self_ * qy_[k];
            // This cell's own share of the holder deflection is solved with
            // it (Gauss-Seidel on contact plus holder compliance).
            float diag = tangential_self_ + series;
            float tx = (cx + series * qx_[k] - sx_[k] - other_x) / diag;
            float ty = (cy + series * qy_[k] - sy_[k] - other_y) / diag;
            float cap = d_.friction * p_[k];
            float mag = std::sqrt(tx * tx + ty * ty);
            stick_[k] = mag <= cap;
            if (!stick_[k]) {
                float s = mag > 0.0f ? cap / mag : 0.0f;
                tx *= s;
                ty *= s;
            }
            float dx = tx - qx_[k], dy = ty - qy_[k];
            if (dx == 0.0f && dy == 0.0f) continue;
            qx_[k] = tx;
            qy_[k] = ty;
            if (coupled_) {
                for (int b = 0; b < nc; b++) {
                    float c = kernel(contact[b], k, true);
                    ux[b] += c * dx;
                    uy[b] += c * dy;
                }
            } else {
                ux[a] += tangential_self_ * dx;
                uy[a] += tangential_self_ * dy;
            }
            cx -= series * dx;
            cy -= series * dy;
            worst = std::max(worst, std::sqrt(dx * dx + dy * dy) * diag);
        }
        if (worst < d_.tolerance) break;
    }

    // Sliding cells: the displacement they could not carry becomes slip.
    for (int a = 0; a < nc; a++) {
        int k = contact[a];
        if (!stick_[k]) {
            sx_[k] = cx - ux[a];
            sy_[k] = cy - uy[a];
        }
    }
}

}  // namespace phys
