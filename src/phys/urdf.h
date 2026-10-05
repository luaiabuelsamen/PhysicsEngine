#pragma once

// URDF import for the rigid solver.
//
// Builds a ModelDesc from a URDF's kinematic tree: one body per link with
// mass (its frame at the link's centre of mass, along the principal axes of
// the URDF inertia), revolute / continuous / prismatic joints as hinges /
// sliders with the URDF limits, and fixed joints either merged away
// (massless links such as tool frames) or as fixed joints. The root link is
// attached to the world. Collision geometry is not imported yet: bodies have
// Shape::None, or with `sphere_radius` > 0 a rough collision sphere at each
// link's centre of mass.

#include <map>
#include <string>
#include <vector>

#include "phys/phys.h"

namespace phys {

struct UrdfOptions {
    Vec3 base_position{0.0f, 0.0f, 0.0f};
    Quat base_rotation;                  // identity
    Vec3 gravity{0.0f, 0.0f, -9.81f};    // URDFs are z-up
    Actuator actuator = Actuator::Position;
    float kp = 1e5f;                     // position actuator gains
    float kd = 1e3f;
    bool effort_limits = true;           // max_force from <limit effort>
    float joint_damping = 0.0f;
    float sphere_radius = 0.0f;          // > 0: rough collision sphere per link
};

struct Pose {
    double p[3] = {0, 0, 0};
    double q[4] = {1, 0, 0, 0};  // w, x, y, z
};

class UrdfModel {
public:
    ModelDesc model;
    std::vector<std::string> body_links;   // libphys body index -> URDF link
    std::vector<std::string> joint_names;  // libphys joint index -> URDF joint (actuated only)
    std::vector<float> lower, upper;       // joint limits (actuated joints)

    int num_dofs() const { return (int)joint_names.size(); }

    // World pose of any URDF link for joint positions q (actuated joints, in
    // joint_names order), by walking the kinematic tree.
    Pose link_pose(const std::string& link, const std::vector<double>& q) const;

    // Write the body states of env `env` for joint positions q (at rest).
    void set_configuration(HostState& state, int env, const std::vector<double>& q) const;

    // Internal tree description (public for tools and bindings).
    struct LinkInfo {
        std::string parent_joint;  // empty for the root
        bool has_mass = false;
        Pose inertial;             // centre-of-mass frame in the link frame
        int body = -1;             // libphys body, or -1 (root / merged)
    };
    struct JointInfo {
        std::string name, type, parent, child;
        Pose origin;
        double axis[3] = {1, 0, 0};
        int dof = -1;              // index into joint_names, or -1 if fixed
    };
    std::map<std::string, LinkInfo> links;
    std::map<std::string, JointInfo> joints;
    std::string root;
    Pose base;
};

// Throws std::runtime_error if the file cannot be read or is not a tree.
UrdfModel load_urdf(const std::string& path, const UrdfOptions& options = UrdfOptions());

}  // namespace phys
