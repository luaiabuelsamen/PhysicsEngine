#pragma once

// Internal interface between World and its CPU / CUDA backends.

#include <memory>
#include <vector>

#include "phys/common.h"
#include "phys/phys.h"

namespace phys {
namespace detail {

// Per-body model arrays, [nbody] (or [nbody * 3]).
struct ModelArrays {
    std::vector<int> shape;
    std::vector<float> size, radius, inv_mass, inv_inertia, restitution, friction;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual void upload(const HostState& state) = 0;
    virtual void download(HostState& state) = 0;
    virtual void step(float dt, int nsteps) = 0;
    virtual void synchronize() = 0;
    virtual void download_contact_counts(std::vector<int>& counts) = 0;
    virtual StateView view() = 0;
};

std::unique_ptr<Backend> make_cpu_backend(const Params& params, const ModelArrays& model);
std::unique_ptr<Backend> make_cuda_backend(const Params& params, const ModelArrays& model);

}  // namespace detail
}  // namespace phys
