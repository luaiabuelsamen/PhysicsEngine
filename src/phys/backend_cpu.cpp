// Single-threaded CPU backend: runs the shared routines in plain loops.

#include <algorithm>
#include <numeric>

#include "phys/backend.h"
#include "phys/particle.h"
#include "phys/rigid.h"

namespace phys {
namespace detail {
namespace {

class CpuBackend final : public Backend {
public:
    CpuBackend(const Params& params, const ModelArrays& model)
        : p_(params), n_(params.nenv * params.nbody), model_(model) {
        HostState initial(n_);
        upload(initial);
        if (p_.rigid) {
            for (int a = 0; a < 3; a++) {
                disp_[a].assign(n_, 0.0f);
                drot_[a].assign(n_, 0.0f);
            }
            contacts_.resize((size_t)p_.nenv * p_.max_contacts);
            ncontact_.assign(p_.nenv, 0);
            size_t nj = (size_t)p_.nenv * p_.njoint;
            ctrl_.assign(nj, 0.0f);
            joint_q_.assign(nj, 0.0f);
            joint_qd_.assign(nj, 0.0f);
            joint_lambda_.assign(nj, 0.0f);
        } else {
            for (int a = 0; a < 3; a++) {
                dpos_[a].assign(n_, 0.0f);
                dvel_[a].assign(n_, 0.0f);
            }
            if (p_.grid) {
                key_.resize(n_);
                sorted_key_.resize(n_);
                sorted_index_.resize(n_);
                cell_start_.resize((size_t)p_.ncell * p_.nenv);
                cell_end_.resize((size_t)p_.ncell * p_.nenv);
            }
        }
    }

    // Copies in place, so the pointers handed out by view() stay valid.
    void upload(const HostState& s) override {
        auto in = [](const auto& src, auto& dst) {
            dst.resize(src.size());
            std::copy(src.begin(), src.end(), dst.begin());
        };
        in(s.px, pos_[0]); in(s.py, pos_[1]); in(s.pz, pos_[2]);
        in(s.vx, vel_[0]); in(s.vy, vel_[1]); in(s.vz, vel_[2]);
        in(s.qw, quat_[0]); in(s.qx, quat_[1]); in(s.qy, quat_[2]); in(s.qz, quat_[3]);
        in(s.wx, angvel_[0]); in(s.wy, angvel_[1]); in(s.wz, angvel_[2]);
        in(s.enabled, enabled_);
    }

    void download(HostState& s) override {
        s.px = pos_[0]; s.py = pos_[1]; s.pz = pos_[2];
        s.vx = vel_[0]; s.vy = vel_[1]; s.vz = vel_[2];
        s.qw = quat_[0]; s.qx = quat_[1]; s.qy = quat_[2]; s.qz = quat_[3];
        s.wx = angvel_[0]; s.wy = angvel_[1]; s.wz = angvel_[2];
        s.enabled = enabled_;
    }

    void step(float dt, int nsteps) override {
        set_step_length(p_, dt);
        Buffers b = buffers();
        for (int step = 0; step < nsteps; step++) {
            if (p_.rigid)
                rigid_step(b);
            else
                particle_step(b);
        }
        if (p_.rigid)
            for (int i = 0; i < p_.nenv * p_.njoint; i++) rigid_observe_joint(p_, b, i);
    }

    void synchronize() override {}

    void download_contact_counts(std::vector<int>& counts) override { counts = ncontact_; }

    void upload_controls(const std::vector<float>& ctrl) override {
        std::copy(ctrl.begin(), ctrl.end(), ctrl_.begin());  // keep StateView pointers valid
    }

    void download_joint_state(std::vector<float>& q, std::vector<float>& qd) override {
        q = joint_q_;
        qd = joint_qd_;
    }

    StateView view() override {
        return {pos_[0].data(),    pos_[1].data(),    pos_[2].data(),
                vel_[0].data(),    vel_[1].data(),    vel_[2].data(),
                quat_[0].data(),   quat_[1].data(),   quat_[2].data(), quat_[3].data(),
                angvel_[0].data(), angvel_[1].data(), angvel_[2].data(),
                enabled_.data(),   ctrl_.data(),      joint_q_.data(), joint_qd_.data(),
                Device::CPU,       p_.nenv,           p_.nbody,        p_.njoint};
    }

private:
    void rigid_step(const Buffers& b) {
        for (int sub = 0; sub < p_.substeps; sub++) {
            if (p_.has_torque_actuators)
                for (int e = 0; e < p_.nenv; e++) rigid_apply_actuators(p_, b, e);
            for (int g = 0; g < n_; g++) rigid_integrate(p_, b, g);
            for (int e = 0; e < p_.nenv; e++) rigid_solve_positions(p_, b, e);
            for (int g = 0; g < n_; g++) rigid_update_velocities(p_, b, g);
            for (int e = 0; e < p_.nenv; e++) rigid_solve_velocities(p_, b, e);
        }
    }

    void particle_step(const Buffers& b) {
        for (int g = 0; g < n_; g++) integrate_body(p_, b, g);

        if (p_.grid) {
            for (int g = 0; g < n_; g++) compute_key(p_, b, g);
            // Stable, so ties keep body order - the same order CUB's radix sort
            // produces on the GPU.
            std::iota(sorted_index_.begin(), sorted_index_.end(), 0);
            std::stable_sort(sorted_index_.begin(), sorted_index_.end(),
                             [this](int x, int y) { return key_[x] < key_[y]; });
            for (int s = 0; s < n_; s++) sorted_key_[s] = key_[sorted_index_[s]];
            std::fill(cell_start_.begin(), cell_start_.end(), -1);
            for (int s = 0; s < n_; s++) find_cell_bounds(p_, b, s, n_);
            for (int g = 0; g < n_; g++) collide_grid(p_, b, g);
        } else {
            for (int g = 0; g < n_; g++) collide_all_pairs(p_, b, g);
        }

        for (int g = 0; g < n_; g++) apply_deltas(b, g);
    }

    Buffers buffers() {
        Buffers b{};
        for (int a = 0; a < 3; a++) {
            b.pos[a] = pos_[a].data();
            b.vel[a] = vel_[a].data();
            b.angvel[a] = angvel_[a].data();
            b.dpos[a] = dpos_[a].data();
            b.dvel[a] = dvel_[a].data();
            b.disp[a] = disp_[a].data();
            b.drot[a] = drot_[a].data();
        }
        for (int a = 0; a < 4; a++) b.quat[a] = quat_[a].data();
        b.enabled = enabled_.data();
        b.shape = model_.shape.data();
        b.size = model_.size.data();
        b.radius = model_.radius.data();
        b.inv_mass = model_.inv_mass.data();
        b.inv_inertia = model_.inv_inertia.data();
        b.restitution = model_.restitution.data();
        b.friction = model_.friction.data();
        b.key = key_.data();
        b.sorted_key = sorted_key_.data();
        b.sorted_index = sorted_index_.data();
        b.cell_start = cell_start_.data();
        b.cell_end = cell_end_.data();
        b.contacts = contacts_.data();
        b.ncontact = ncontact_.data();
        b.joints = model_.joints.data();
        b.may_collide = model_.may_collide.data();
        b.ctrl = ctrl_.data();
        b.joint_q = joint_q_.data();
        b.joint_qd = joint_qd_.data();
        b.joint_lambda = joint_lambda_.data();
        return b;
    }

    Params p_;
    int n_;
    ModelArrays model_;
    std::vector<float> pos_[3], vel_[3], quat_[4], angvel_[3];
    std::vector<uint8_t> enabled_;
    // Particle scratch
    std::vector<float> dpos_[3], dvel_[3];
    std::vector<int> key_, sorted_key_, sorted_index_, cell_start_, cell_end_;
    // Rigid scratch
    std::vector<float> disp_[3], drot_[3];
    std::vector<Contact> contacts_;
    std::vector<int> ncontact_;
    std::vector<float> ctrl_, joint_q_, joint_qd_, joint_lambda_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const Params& params, const ModelArrays& model) {
    return std::unique_ptr<Backend>(new CpuBackend(params, model));
}

}  // namespace detail
}  // namespace phys
