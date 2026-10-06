// CUDA backend. Per-body phases run one thread per body; the rigid solver's
// per-env phases run one thread per env. All buffers, including the sort's
// scratch space, are allocated once, so a step is a fixed sequence of kernel
// launches with no host round-trips.

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <stdexcept>
#include <string>

#include "phys/backend.h"
#include "phys/particle.h"
#include "phys/rigid.h"
#include "phys/tactile_sensor.h"

namespace phys {
namespace detail {
namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("phys CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
}
#define PHYS_CUDA(call) check((call), #call)

constexpr int kThreads = 256;
// Per-env kernels: smaller blocks spread few envs across more SMs.
constexpr int kEnvThreads = 64;
int blocks_for(int n, int threads = kThreads) { return (n + threads - 1) / threads; }

__device__ __forceinline__ int thread_index() { return blockIdx.x * blockDim.x + threadIdx.x; }

// --- particle solver ---------------------------------------------------------

__global__ void integrate_kernel(Params p, Buffers b, int n) {
    int g = thread_index();
    if (g < n) integrate_body(p, b, g);
}

__global__ void key_kernel(Params p, Buffers b, int* index, int n) {
    int g = thread_index();
    if (g >= n) return;
    compute_key(p, b, g);
    index[g] = g;
}

__global__ void cell_bounds_kernel(Params p, Buffers b, int n) {
    int s = thread_index();
    if (s < n) find_cell_bounds(p, b, s, n);
}

// One thread per sorted slot, so neighbouring threads handle nearby bodies.
__global__ __launch_bounds__(kThreads) void collide_grid_kernel(Params p, Buffers b, int n) {
    int s = thread_index();
    if (s < n) collide_grid(p, b, b.sorted_index[s]);
}

__global__ void collide_all_pairs_kernel(Params p, Buffers b, int n) {
    int g = thread_index();
    if (g < n) collide_all_pairs(p, b, g);
}

__global__ void apply_kernel(Buffers b, int n) {
    int g = thread_index();
    if (g < n) apply_deltas(b, g);
}

// --- rigid solver ------------------------------------------------------------

__global__ void rigid_actuators_kernel(Params p, Buffers b) {
    int e = thread_index();
    if (e < p.nenv) rigid_apply_actuators(p, b, e);
}

__global__ void rigid_observe_kernel(Params p, Buffers b, int n) {
    int i = thread_index();
    if (i < n) rigid_observe_joint(p, b, i);
}

__global__ void rigid_integrate_kernel(Params p, Buffers b, int n) {
    int g = thread_index();
    if (g < n) rigid_integrate(p, b, g);
}

__global__ void rigid_positions_kernel(Params p, Buffers b) {
    int e = thread_index();
    if (e < p.nenv) rigid_solve_positions(p, b, e);
}

__global__ void rigid_update_velocities_kernel(Params p, Buffers b, int n) {
    int g = thread_index();
    if (g < n) rigid_update_velocities(p, b, g);
}

__global__ void rigid_velocities_kernel(Params p, Buffers b) {
    int e = thread_index();
    if (e < p.nenv) rigid_solve_velocities(p, b, e);
}

__global__ void tactile_accumulate_kernel(Params p, Buffers b, bool first) {
    int i = thread_index();
    if (i < p.nenv * p.nsensor) tactile_accumulate(p, b, i % p.nenv, i / p.nenv, first);
}

// One block per (env, sensor), threads over the sensor's cells. Shared
// memory holds the cell lists and influence table, and - when
// p.tactile_shared_work - the working arrays too.
__global__ __launch_bounds__(kTactileMaxThreads) void tactile_kernel(Params p, Buffers b) {
    extern __shared__ float shared[];
    const int n = p.max_sensor_cells;
    BlockExec ex;
    ex.red = shared;
    ex.counts = reinterpret_cast<int*>(shared + kTactileMaxThreads / kWarp);
    float* rest = shared + 2 * (kTactileMaxThreads / kWarp);
    ex.list = reinterpret_cast<int*>(rest);
    ex.vals = rest + n;
    ex.kernel = rest + 2 * n;
    ex.work = p.tactile_shared_work ? rest + 3 * n : nullptr;
    ex.cand = p.tactile_shared_work ? reinterpret_cast<int*>(rest + (3 + kTactileArrays - 1) * n) : nullptr;
    tactile_update(ex, p, b, blockIdx.x % p.nenv, blockIdx.x / p.nenv);
}

size_t tactile_shared_bytes(const Params& p) {
    size_t arrays = p.tactile_shared_work ? 3 + (kTactileArrays - 1) + 1 : 3;
    return (arrays * (size_t)p.max_sensor_cells + 2 * (kTactileMaxThreads / kWarp)) * sizeof(float);
}

// Keep the working arrays in shared memory when they fit comfortably.
constexpr size_t kTactileSharedLimit = 96 * 1024;

// Frees everything it holds, including when a constructor throws part way.
struct DeviceAllocations {
    std::vector<void*> ptrs;
    ~DeviceAllocations() {
        for (void* ptr : ptrs) cudaFree(ptr);
    }
};

class CudaBackend final : public Backend {
public:
    CudaBackend(const Params& params, const ModelArrays& model)
        : p_(params), n_(params.nenv * params.nbody) {
        size_t fb = n_ * sizeof(float);
        for (int a = 0; a < 3; a++) {
            alloc(&b_.pos[a], fb);
            alloc(&b_.vel[a], fb);
            alloc(&b_.angvel[a], fb);
        }
        for (int a = 0; a < 4; a++) alloc(&b_.quat[a], fb);
        alloc(&b_.enabled, n_);
        upload(HostState(n_));

        b_.shape = upload_model(model.shape);
        b_.size = upload_model(model.size);
        b_.radius = upload_model(model.radius);
        b_.inv_mass = upload_model(model.inv_mass);
        b_.inv_inertia = upload_model(model.inv_inertia);
        b_.restitution = upload_model(model.restitution);
        b_.friction = upload_model(model.friction);

        if (p_.rigid) {
            for (int a = 0; a < 3; a++) {
                alloc(&b_.disp[a], fb);
                alloc(&b_.drot[a], fb);
            }
            alloc(&b_.contacts, (size_t)p_.nenv * p_.max_contacts * sizeof(Contact));
            alloc(&b_.ncontact, p_.nenv * sizeof(int));
            PHYS_CUDA(cudaMemset(b_.ncontact, 0, p_.nenv * sizeof(int)));
            b_.joints = upload_model(model.joints);
            b_.may_collide = upload_model(model.may_collide);
            size_t jb = (size_t)p_.nenv * p_.njoint * sizeof(float);
            for (float** buf : {&b_.ctrl, &b_.joint_q, &b_.joint_qd, &b_.joint_lambda}) {
                alloc(buf, jb);
                PHYS_CUDA(cudaMemset(*buf, 0, jb));
            }
            b_.sensors = upload_model(model.sensors);
            b_.tactile_kernel = upload_model(model.tactile_kernel);
            size_t tb = (size_t)p_.nenv * p_.ntactile * kTactileChannels * sizeof(float);
            size_t fb3 = (size_t)p_.nenv * p_.nsensor * 3 * sizeof(float);
            alloc(&b_.tactile, tb);
            alloc(&b_.tactile_force, fb3);
            PHYS_CUDA(cudaMemset(b_.tactile, 0, tb));
            PHYS_CUDA(cudaMemset(b_.tactile_force, 0, fb3));
            alloc(&b_.tactile_scratch, model.tactile_scratch * sizeof(float));
            PHYS_CUDA(cudaMemset(b_.tactile_scratch, 0, model.tactile_scratch * sizeof(float)));
            p_.tactile_shared_work = false;
            if (p_.nsensor > 0) {
                Params trial = p_;
                trial.tactile_shared_work = true;
                p_.tactile_shared_work = tactile_shared_bytes(trial) <= kTactileSharedLimit;
            }
            if (p_.nsensor > 0)
                PHYS_CUDA(cudaFuncSetAttribute(tactile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               (int)tactile_shared_bytes(p_)));
            alloc(&b_.tactile_list, model.tactile_list * sizeof(int));
            alloc(&b_.tactile_accum, (size_t)p_.nenv * p_.nsensor * kTactileAccum * sizeof(float));
            alloc(&b_.tactile_touch, (size_t)p_.nenv * p_.nsensor * kTactileMaxTouching * sizeof(int));
            PHYS_CUDA(cudaMemset(b_.tactile_accum, 0, (size_t)p_.nenv * p_.nsensor * kTactileAccum * sizeof(float)));
            PHYS_CUDA(cudaMemset(b_.tactile_touch, 0xff, (size_t)p_.nenv * p_.nsensor * kTactileMaxTouching * sizeof(int)));
            size_t cb = (size_t)p_.nenv * p_.nsensor * kPadState * sizeof(float);
            alloc(&b_.tactile_coupling, cb);
            PHYS_CUDA(cudaMemset(b_.tactile_coupling, 0, cb));
        } else {
            for (int a = 0; a < 3; a++) {
                alloc(&b_.dpos[a], fb);
                alloc(&b_.dvel[a], fb);
            }
            if (p_.grid) init_grid();
        }
    }

    void upload(const HostState& s) override {
        copy_in(b_.pos[0], s.px); copy_in(b_.pos[1], s.py); copy_in(b_.pos[2], s.pz);
        copy_in(b_.vel[0], s.vx); copy_in(b_.vel[1], s.vy); copy_in(b_.vel[2], s.vz);
        copy_in(b_.quat[0], s.qw); copy_in(b_.quat[1], s.qx);
        copy_in(b_.quat[2], s.qy); copy_in(b_.quat[3], s.qz);
        copy_in(b_.angvel[0], s.wx); copy_in(b_.angvel[1], s.wy); copy_in(b_.angvel[2], s.wz);
        copy_in(b_.enabled, s.enabled);
    }

    void download(HostState& s) override {
        copy_out(s.px, b_.pos[0]); copy_out(s.py, b_.pos[1]); copy_out(s.pz, b_.pos[2]);
        copy_out(s.vx, b_.vel[0]); copy_out(s.vy, b_.vel[1]); copy_out(s.vz, b_.vel[2]);
        copy_out(s.qw, b_.quat[0]); copy_out(s.qx, b_.quat[1]);
        copy_out(s.qy, b_.quat[2]); copy_out(s.qz, b_.quat[3]);
        copy_out(s.wx, b_.angvel[0]); copy_out(s.wy, b_.angvel[1]); copy_out(s.wz, b_.angvel[2]);
        copy_out(s.enabled, b_.enabled);
    }

    void step(float dt, int nsteps) override {
        set_step_length(p_, dt);
        int ns = p_.nenv * p_.nsensor;
        for (int step = 0; step < nsteps; step++) {
            if (p_.rigid) {
                rigid_step();
                // Every step: coupled pads take their stiffness from it.
                if (ns > 0) tactile_kernel<<<ns, p_.tactile_threads, tactile_shared_bytes(p_)>>>(p_, b_);
            } else {
                particle_step();
            }
        }
        int nj = p_.nenv * p_.njoint;
        if (p_.rigid && nj > 0) rigid_observe_kernel<<<blocks_for(nj), kThreads>>>(p_, b_, nj);
        PHYS_CUDA(cudaGetLastError());
    }

    void synchronize() override { PHYS_CUDA(cudaDeviceSynchronize()); }

    void download_contact_counts(std::vector<int>& counts) override {
        counts.resize(p_.nenv);
        copy_out(counts, b_.ncontact);
    }

    void upload_controls(const std::vector<float>& ctrl) override { copy_in(b_.ctrl, ctrl); }

    void download_joint_state(std::vector<float>& q, std::vector<float>& qd) override {
        copy_out(q, b_.joint_q);
        copy_out(qd, b_.joint_qd);
    }

    void download_tactile(std::vector<float>& cells, std::vector<float>& forces) override {
        copy_out(cells, b_.tactile);
        copy_out(forces, b_.tactile_force);
    }

    StateView view() override {
        return {b_.pos[0],    b_.pos[1],    b_.pos[2],    b_.vel[0],     b_.vel[1],
                b_.vel[2],    b_.quat[0],   b_.quat[1],   b_.quat[2],    b_.quat[3],
                b_.angvel[0], b_.angvel[1], b_.angvel[2], b_.enabled,    b_.ctrl,
                b_.joint_q,   b_.joint_qd,  b_.tactile,   b_.tactile_force,
                Device::CUDA, p_.nenv,      p_.nbody,     p_.njoint,     p_.nsensor,
                p_.ntactile};
    }

private:
    void rigid_step() {
        int body_blocks = blocks_for(n_);
        int env_blocks = blocks_for(p_.nenv, kEnvThreads);
        for (int sub = 0; sub < p_.substeps; sub++) {
            if (p_.has_torque_actuators)
                rigid_actuators_kernel<<<env_blocks, kEnvThreads>>>(p_, b_);
            rigid_integrate_kernel<<<body_blocks, kThreads>>>(p_, b_, n_);
            rigid_positions_kernel<<<env_blocks, kEnvThreads>>>(p_, b_);
            rigid_update_velocities_kernel<<<body_blocks, kThreads>>>(p_, b_, n_);
            rigid_velocities_kernel<<<env_blocks, kEnvThreads>>>(p_, b_);
            int ns = p_.nenv * p_.nsensor;
            if (ns > 0)
                tactile_accumulate_kernel<<<blocks_for(ns, kEnvThreads), kEnvThreads>>>(p_, b_, sub == 0);
        }
    }

    void particle_step() {
        int blocks = blocks_for(n_);
        integrate_kernel<<<blocks, kThreads>>>(p_, b_, n_);

        if (p_.grid) {
            key_kernel<<<blocks, kThreads>>>(p_, b_, index_, n_);
            // LSD radix sort is stable, matching the CPU backend's order.
            PHYS_CUDA(cub::DeviceRadixSort::SortPairs(
                sort_temp_, sort_bytes_, b_.key, sorted_key_, index_, sorted_index_, n_, 0,
                end_bit_));
            PHYS_CUDA(cudaMemsetAsync(b_.cell_start, 0xff, cell_bytes_));  // -1
            cell_bounds_kernel<<<blocks, kThreads>>>(p_, b_, n_);
            collide_grid_kernel<<<blocks, kThreads>>>(p_, b_, n_);
        } else {
            collide_all_pairs_kernel<<<blocks, kThreads>>>(p_, b_, n_);
        }

        apply_kernel<<<blocks, kThreads>>>(b_, n_);
    }

    void init_grid() {
        size_t ib = n_ * sizeof(int);
        size_t cb = (size_t)p_.ncell * p_.nenv * sizeof(int);
        alloc(&b_.key, ib);
        alloc(&index_, ib);
        alloc(&sorted_key_, ib);
        alloc(&sorted_index_, ib);
        alloc(&b_.cell_start, cb);
        alloc(&b_.cell_end, cb);
        b_.sorted_key = sorted_key_;
        b_.sorted_index = sorted_index_;
        cell_bytes_ = cb;

        // Only sort the bits a key can use; the sentinel is the largest key.
        while (end_bit_ < 31 && (1 << end_bit_) <= p_.sentinel_key) end_bit_++;
        PHYS_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes_, b_.key, sorted_key_,
                                                  index_, sorted_index_, n_, 0, end_bit_));
        alloc(&sort_temp_, sort_bytes_);
    }

    template <typename T>
    void alloc(T** ptr, size_t bytes) {
        void* raw = nullptr;
        PHYS_CUDA(cudaMalloc(&raw, bytes > 0 ? bytes : 1));
        allocations_.ptrs.push_back(raw);
        *ptr = static_cast<T*>(raw);
    }

    template <typename T>
    const T* upload_model(const std::vector<T>& src) {
        T* ptr;
        alloc(&ptr, src.size() * sizeof(T));
        copy_in(ptr, src);
        return ptr;
    }

    template <typename T>
    void copy_in(T* dst, const std::vector<T>& src) {
        PHYS_CUDA(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    template <typename T>
    void copy_out(std::vector<T>& dst, const T* src) {
        PHYS_CUDA(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }

    Params p_;
    int n_;
    Buffers b_{};
    int* index_ = nullptr;
    int* sorted_key_ = nullptr;
    int* sorted_index_ = nullptr;
    void* sort_temp_ = nullptr;
    size_t sort_bytes_ = 0;
    size_t cell_bytes_ = 0;
    int end_bit_ = 1;
    DeviceAllocations allocations_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(const Params& params, const ModelArrays& model) {
    return std::unique_ptr<Backend>(new CudaBackend(params, model));
}

}  // namespace detail
}  // namespace phys
