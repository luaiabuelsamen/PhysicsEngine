// C interface to phys::ElasticPatch, for driving it from Python (ctypes) in
// validation and calibration scripts (tools/sparsh_validate.py).

#include <exception>

#include "phys/tactile.h"

extern "C" {

// Replays an indenter trajectory on a fresh patch.
//   patch: width, height, cell, dome_radius, youngs_modulus, poisson, friction,
//          tangential_compliance_scale, model (0 half-space, 1 winkler),
//          winkler_stiffness, winkler_shear_ratio, series_tangential_stiffness
//   indenter: kind (0 sphere, 1 flat punch, 2 cone), radius, half_angle
//   tips: n x 3 indenter tip positions (patch frame, metres)
//   forces: n x 3 output, (shear x, shear y, normal) in newtons
// Returns 0 on success, -1 on error.
int phys_patch_replay(const float* patch, const float* indenter, const float* tips, int n,
                      float* forces) {
    try {
        phys::ElasticPatchDesc d;
        d.width = patch[0];
        d.height = patch[1];
        d.cell = patch[2];
        d.dome_radius = patch[3];
        d.youngs_modulus = patch[4];
        d.poisson = patch[5];
        d.friction = patch[6];
        d.tangential_compliance_scale = patch[7];
        d.model = (phys::ElasticPatchDesc::Model)(int)patch[8];
        d.winkler_stiffness = patch[9];
        d.winkler_shear_ratio = patch[10];
        d.series_tangential_stiffness = patch[11];
        d.max_iterations = 300;
        d.tolerance = 1e-9f;
        phys::ElasticPatch p(d);
        phys::Indenter ind;
        ind.kind = (phys::Indenter::Kind)(int)indenter[0];
        ind.radius = indenter[1];
        ind.half_angle = indenter[2];
        for (int i = 0; i < n; i++) {
            p.step(ind, {tips[3 * i], tips[3 * i + 1], tips[3 * i + 2]});
            phys::Vec3 f = p.force();
            forces[3 * i] = f.x;
            forces[3 * i + 1] = f.y;
            forces[3 * i + 2] = f.z;
        }
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

}  // extern "C"
