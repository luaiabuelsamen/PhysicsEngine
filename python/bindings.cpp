// Python bindings for libphys (module libphys._libphys).
//
// Model descriptions, World and ElasticPatch are bound directly. World's
// state buffers are exposed as raw pointers (World._buffers); the Python
// package wraps them as zero-copy torch tensors (python/libphys/__init__.py).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <sstream>

#include "phys/phys.h"
#include "phys/tactile.h"
#ifdef PHYS_HAS_URDF
#include "phys/urdf.h"
#endif
#ifdef PHYS_HAS_PLANNER
#include <pybind11/eigen.h>

#include "robot.hpp"
#endif

namespace py = pybind11;
using namespace phys;

namespace {

Vec3 vec3_from(py::sequence s) {
    if (py::len(s) != 3) throw py::value_error("expected 3 values");
    return {s[0].cast<float>(), s[1].cast<float>(), s[2].cast<float>()};
}

Quat quat_from(py::sequence s) {
    if (py::len(s) != 4) throw py::value_error("expected 4 values (w, x, y, z)");
    Quat q;
    q.w = s[0].cast<float>();
    q.x = s[1].cast<float>();
    q.y = s[2].cast<float>();
    q.z = s[3].cast<float>();
    return q;
}

// Raw buffer table: name -> (address, shape, numpy typestr).
py::dict buffers(World& w) {
    StateView v = w.state();
    auto entry = [&](const void* ptr, int cols, const char* type) {
        return py::make_tuple((std::uintptr_t)ptr, py::make_tuple(v.nenv, cols), type);
    };
    py::dict d;
    const char* f = "<f4";
    d["px"] = entry(v.px, v.nbody, f); d["py"] = entry(v.py, v.nbody, f); d["pz"] = entry(v.pz, v.nbody, f);
    d["vx"] = entry(v.vx, v.nbody, f); d["vy"] = entry(v.vy, v.nbody, f); d["vz"] = entry(v.vz, v.nbody, f);
    d["qw"] = entry(v.qw, v.nbody, f); d["qx"] = entry(v.qx, v.nbody, f);
    d["qy"] = entry(v.qy, v.nbody, f); d["qz"] = entry(v.qz, v.nbody, f);
    d["wx"] = entry(v.wx, v.nbody, f); d["wy"] = entry(v.wy, v.nbody, f); d["wz"] = entry(v.wz, v.nbody, f);
    d["enabled"] = entry(v.enabled, v.nbody, "|u1");
    if (v.njoint > 0) {
        d["ctrl"] = entry(v.ctrl, v.njoint, f);
        d["joint_q"] = entry(v.joint_q, v.njoint, f);
        d["joint_qd"] = entry(v.joint_qd, v.njoint, f);
    }
    if (v.nsensor > 0) {
        d["tactile"] = entry(v.tactile, v.ntactile * World::kTactileChannels, f);
        d["tactile_force"] = entry(v.tactile_force, v.nsensor * 3, f);
    }
    return d;
}

}  // namespace

PYBIND11_MODULE(_libphys, m) {
    m.doc() = "libphys: batched rigid-body and tactile contact simulation (C++ / CUDA core)";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init([](float x, float y, float z) { return Vec3{x, y, z}; }), py::arg("x") = 0.0f,
             py::arg("y") = 0.0f, py::arg("z") = 0.0f)
        .def(py::init(&vec3_from))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("__iter__", [](const Vec3& v) { return py::iter(py::make_tuple(v.x, v.y, v.z)); })
        .def("__repr__", [](const Vec3& v) {
            std::ostringstream os;
            os << "Vec3(" << v.x << ", " << v.y << ", " << v.z << ")";
            return os.str();
        });
    py::implicitly_convertible<py::tuple, Vec3>();
    py::implicitly_convertible<py::list, Vec3>();

    py::class_<Quat>(m, "Quat")
        .def(py::init<>())
        .def(py::init(&quat_from))
        .def_readwrite("w", &Quat::w)
        .def_readwrite("x", &Quat::x)
        .def_readwrite("y", &Quat::y)
        .def_readwrite("z", &Quat::z)
        .def("__iter__", [](const Quat& q) { return py::iter(py::make_tuple(q.w, q.x, q.y, q.z)); });
    py::implicitly_convertible<py::tuple, Quat>();
    py::implicitly_convertible<py::list, Quat>();

    py::enum_<Device>(m, "Device").value("CPU", Device::CPU).value("CUDA", Device::CUDA);
    py::enum_<Solver>(m, "Solver").value("Rigid", Solver::Rigid).value("Particle", Solver::Particle);
    py::enum_<Broadphase>(m, "Broadphase")
        .value("Auto", Broadphase::Auto)
        .value("AllPairs", Broadphase::AllPairs)
        .value("Grid", Broadphase::Grid);
    py::enum_<Shape>(m, "Shape")
        .value("Sphere", Shape::Sphere)
        .value("Capsule", Shape::Capsule)
        .value("Box", Shape::Box)
        .value("Plane", Shape::Plane)
        .value("None_", Shape::None);
    py::enum_<JointType>(m, "JointType")
        .value("Hinge", JointType::Hinge)
        .value("Slider", JointType::Slider)
        .value("Ball", JointType::Ball)
        .value("Fixed", JointType::Fixed);
    py::enum_<Actuator>(m, "Actuator")
        .value("None_", Actuator::None)
        .value("Torque", Actuator::Torque)
        .value("Position", Actuator::Position)
        .value("Velocity", Actuator::Velocity);

    py::class_<BodyDesc>(m, "BodyDesc")
        .def(py::init<>())
        .def_readwrite("shape", &BodyDesc::shape)
        .def_readwrite("size", &BodyDesc::size)
        .def_readwrite("mass", &BodyDesc::mass)
        .def_readwrite("restitution", &BodyDesc::restitution)
        .def_readwrite("friction", &BodyDesc::friction)
        .def_readwrite("inertia", &BodyDesc::inertia)
        .def_readwrite("collision_group", &BodyDesc::collision_group)
        .def_readwrite("collision_mask", &BodyDesc::collision_mask)
        .def_static("sphere", &BodyDesc::sphere, py::arg("radius"), py::arg("mass"))
        .def_static("capsule", &BodyDesc::capsule, py::arg("radius"), py::arg("half_length"), py::arg("mass"))
        .def_static("box", &BodyDesc::box, py::arg("half_extents"), py::arg("mass"))
        .def_static("plane", &BodyDesc::plane)
        .def_static("none", &BodyDesc::none, py::arg("mass"), py::arg("inertia"));

    py::class_<JointDesc>(m, "JointDesc")
        .def(py::init<>())
        .def_readwrite("type", &JointDesc::type)
        .def_readwrite("parent", &JointDesc::parent)
        .def_readwrite("child", &JointDesc::child)
        .def_readwrite("parent_anchor", &JointDesc::parent_anchor)
        .def_readwrite("parent_frame", &JointDesc::parent_frame)
        .def_readwrite("child_anchor", &JointDesc::child_anchor)
        .def_readwrite("child_frame", &JointDesc::child_frame)
        .def_readwrite("limited", &JointDesc::limited)
        .def_readwrite("lower", &JointDesc::lower)
        .def_readwrite("upper", &JointDesc::upper)
        .def_readwrite("damping", &JointDesc::damping)
        .def_readwrite("actuator", &JointDesc::actuator)
        .def_readwrite("kp", &JointDesc::kp)
        .def_readwrite("kd", &JointDesc::kd)
        .def_readwrite("max_force", &JointDesc::max_force)
        .def_static("hinge", &JointDesc::hinge, py::arg("parent"), py::arg("child"),
                    py::arg("parent_anchor"), py::arg("child_anchor"), py::arg("axis"))
        .def_static("slider", &JointDesc::slider, py::arg("parent"), py::arg("child"),
                    py::arg("parent_anchor"), py::arg("child_anchor"), py::arg("axis"))
        .def_static("ball", &JointDesc::ball, py::arg("parent"), py::arg("child"),
                    py::arg("parent_anchor"), py::arg("child_anchor"))
        .def_static("fixed", &JointDesc::fixed, py::arg("parent"), py::arg("child"),
                    py::arg("parent_anchor"), py::arg("child_anchor"));

    py::class_<TactileSensorDesc>(m, "TactileSensorDesc")
        .def(py::init<>())
        .def_readwrite("body", &TactileSensorDesc::body)
        .def_readwrite("origin", &TactileSensorDesc::origin)
        .def_readwrite("frame", &TactileSensorDesc::frame)
        .def_readwrite("width", &TactileSensorDesc::width)
        .def_readwrite("height", &TactileSensorDesc::height)
        .def_readwrite("nx", &TactileSensorDesc::nx)
        .def_readwrite("ny", &TactileSensorDesc::ny)
        .def_readwrite("youngs_modulus", &TactileSensorDesc::youngs_modulus)
        .def_readwrite("poisson", &TactileSensorDesc::poisson)
        .def_readwrite("dome_radius", &TactileSensorDesc::dome_radius)
        .def_readwrite("thickness", &TactileSensorDesc::thickness)
        .def_readwrite("max_iterations", &TactileSensorDesc::max_iterations)
        .def_readwrite("tolerance", &TactileSensorDesc::tolerance)
        .def_readwrite("coupled", &TactileSensorDesc::coupled);

    py::class_<ModelDesc>(m, "ModelDesc")
        .def(py::init<>())
        .def_readwrite("bodies", &ModelDesc::bodies)
        .def_readwrite("joints", &ModelDesc::joints)
        .def_readwrite("collide_jointed_bodies", &ModelDesc::collide_jointed_bodies)
        .def_readwrite("gravity", &ModelDesc::gravity)
        .def_readwrite("solver", &ModelDesc::solver)
        .def_readwrite("substeps", &ModelDesc::substeps)
        .def_readwrite("position_iterations", &ModelDesc::position_iterations)
        .def_readwrite("velocity_iterations", &ModelDesc::velocity_iterations)
        .def_readwrite("max_contacts_per_env", &ModelDesc::max_contacts_per_env)
        .def_readwrite("tactile_sensors", &ModelDesc::tactile_sensors)
        .def_readwrite("bounds_lo", &ModelDesc::bounds_lo)
        .def_readwrite("bounds_hi", &ModelDesc::bounds_hi)
        .def_readwrite("wall_restitution", &ModelDesc::wall_restitution)
        .def_readwrite("contact_correction", &ModelDesc::contact_correction)
        .def_readwrite("broadphase", &ModelDesc::broadphase);

    py::class_<World>(m, "World")
        .def(py::init<const ModelDesc&, int, Device>(), py::arg("model"), py::arg("nenv"), py::arg("device"))
        .def_property_readonly("nenv", &World::nenv)
        .def_property_readonly("nbody", &World::nbody)
        .def_property_readonly("njoint", &World::njoint)
        .def_property_readonly("device", &World::device)
        .def_property_readonly("solver", &World::solver)
        .def_property_readonly("max_contacts_per_env", &World::max_contacts_per_env)
        .def_property_readonly("nsensor", &World::nsensor)
        .def("sensor", &World::sensor, py::arg("index"))
        .def("step", &World::step, py::arg("dt"), py::arg("nsteps") = 1,
             py::call_guard<py::gil_scoped_release>())
        .def("synchronize", &World::synchronize, py::call_guard<py::gil_scoped_release>())
        .def("contact_counts", [](World& w) {
            std::vector<int> c;
            w.get_contact_counts(c);
            return py::array_t<int>((py::ssize_t)c.size(), c.data());
        })
        .def("_buffers", &buffers);

    // --- tactile ---
    py::class_<ElasticPatchDesc> pd(m, "ElasticPatchDesc");
    py::enum_<ElasticPatchDesc::Model>(pd, "Model")
        .value("HalfSpace", ElasticPatchDesc::HalfSpace)
        .value("Winkler", ElasticPatchDesc::Winkler);
    pd.def(py::init<>())
        .def_readwrite("model", &ElasticPatchDesc::model)
        .def_readwrite("width", &ElasticPatchDesc::width)
        .def_readwrite("height", &ElasticPatchDesc::height)
        .def_readwrite("cell", &ElasticPatchDesc::cell)
        .def_readwrite("dome_radius", &ElasticPatchDesc::dome_radius)
        .def_readwrite("youngs_modulus", &ElasticPatchDesc::youngs_modulus)
        .def_readwrite("poisson", &ElasticPatchDesc::poisson)
        .def_readwrite("friction", &ElasticPatchDesc::friction)
        .def_readwrite("tangential_compliance_scale", &ElasticPatchDesc::tangential_compliance_scale)
        .def_readwrite("winkler_stiffness", &ElasticPatchDesc::winkler_stiffness)
        .def_readwrite("winkler_shear_ratio", &ElasticPatchDesc::winkler_shear_ratio)
        .def_readwrite("series_tangential_stiffness", &ElasticPatchDesc::series_tangential_stiffness)
        .def_readwrite("max_iterations", &ElasticPatchDesc::max_iterations)
        .def_readwrite("tolerance", &ElasticPatchDesc::tolerance);

    py::class_<Indenter> ind(m, "Indenter");
    py::enum_<Indenter::Kind>(ind, "Kind")
        .value("Sphere", Indenter::Sphere)
        .value("FlatPunch", Indenter::FlatPunch)
        .value("Cone", Indenter::Cone);
    ind.def(py::init<>())
        .def_readwrite("kind", &Indenter::kind)
        .def_readwrite("radius", &Indenter::radius)
        .def_readwrite("half_angle", &Indenter::half_angle);

    py::class_<ElasticPatch>(m, "ElasticPatch")
        .def(py::init<const ElasticPatchDesc&>(), py::arg("desc"))
        .def("reset", &ElasticPatch::reset)
        .def("step", &ElasticPatch::step, py::arg("indenter"), py::arg("tip"))
        .def("force", [](const ElasticPatch& p) {
            Vec3 f = p.force();
            return py::make_tuple(f.x, f.y, f.z);
        })
        .def_property_readonly("cells", &ElasticPatch::cells)
        .def("cell_positions", [](const ElasticPatch& p) {
            py::array_t<float> a({p.cells(), 2});
            auto r = a.mutable_unchecked<2>();
            for (int i = 0; i < p.cells(); i++) {
                r(i, 0) = p.cell_x(i);
                r(i, 1) = p.cell_y(i);
            }
            return a;
        })
        .def("tractions", [](const ElasticPatch& p) {
            py::array_t<float> a({p.cells(), 3});
            auto r = a.mutable_unchecked<2>();
            for (int i = 0; i < p.cells(); i++) {
                Vec3 t = p.traction(i);
                r(i, 0) = t.x;
                r(i, 1) = t.y;
                r(i, 2) = t.z;
            }
            return a;
        }, "Per-cell traction (shear x, shear y, pressure) in Pa")
        .def("sticking", [](const ElasticPatch& p) {
            py::array_t<bool> a(p.cells());
            auto r = a.mutable_unchecked<1>();
            for (int i = 0; i < p.cells(); i++) r(i) = p.sticking(i) && p.pressure(i) > 0.0f;
            return a;
        })
        .def("contact_cells", &ElasticPatch::contact_cells);

#ifdef PHYS_HAS_URDF
    py::class_<UrdfOptions>(m, "UrdfOptions")
        .def(py::init<>())
        .def_readwrite("base_position", &UrdfOptions::base_position)
        .def_readwrite("base_rotation", &UrdfOptions::base_rotation)
        .def_readwrite("gravity", &UrdfOptions::gravity)
        .def_readwrite("actuator", &UrdfOptions::actuator)
        .def_readwrite("kp", &UrdfOptions::kp)
        .def_readwrite("kd", &UrdfOptions::kd)
        .def_readwrite("effort_limits", &UrdfOptions::effort_limits)
        .def_readwrite("joint_damping", &UrdfOptions::joint_damping)
        .def_readwrite("sphere_radius", &UrdfOptions::sphere_radius);

    py::class_<UrdfModel>(m, "UrdfModel")
        .def_readwrite("model", &UrdfModel::model)
        .def_readonly("body_links", &UrdfModel::body_links)
        .def_readonly("joint_names", &UrdfModel::joint_names)
        .def_readonly("lower", &UrdfModel::lower)
        .def_readonly("upper", &UrdfModel::upper)
        .def_property_readonly("num_dofs", &UrdfModel::num_dofs)
        .def("link_pose", [](const UrdfModel& u, const std::string& link, const std::vector<double>& q) {
            Pose p = u.link_pose(link, q);
            return py::make_tuple(py::make_tuple(p.p[0], p.p[1], p.p[2]),
                                  py::make_tuple(p.q[0], p.q[1], p.q[2], p.q[3]));
        }, py::arg("link"), py::arg("q"), "World pose ((x, y, z), (w, x, y, z)) of a link for joint positions q")
        .def("body_poses", [](const UrdfModel& u, const std::vector<double>& q) {
            // [num_bodies, 7]: position and quaternion (w, x, y, z) of every body at rest at q.
            HostState s((int)u.model.bodies.size());
            u.set_configuration(s, 0, q);
            py::array_t<float> a({s.size(), 7});
            auto r = a.mutable_unchecked<2>();
            for (int i = 0; i < s.size(); i++) {
                float v[7] = {s.px[i], s.py[i], s.pz[i], s.qw[i], s.qx[i], s.qy[i], s.qz[i]};
                for (int k = 0; k < 7; k++) r(i, k) = v[k];
            }
            return a;
        }, py::arg("q"));
    m.def("load_urdf", &load_urdf, py::arg("path"), py::arg("options") = UrdfOptions());
#endif

#ifdef PHYS_HAS_PLANNER
    // The C++ motion planner (extern/MotionPlanning): FK / IK / MoveJ / MoveL.
    py::class_<Robot>(m, "Planner")
        .def(py::init([](const std::string& urdf, double dt, const std::string& tip_link) {
                 auto r = new Robot(urdf, tip_link);
                 OptimizerConfig cfg = r->getOptimizerConfig();
                 cfg.dt = dt;
                 r->setOptimizerConfig(cfg);
                 return r;
             }),
             py::arg("urdf"), py::arg("dt") = 0.01, py::arg("tip_link") = "")
        .def_property_readonly("dof", &Robot::getDOF)
        .def_property_readonly("joint_names", &Robot::getJointNames)
        .def("fk", [](Robot& r, const std::vector<double>& q) {
            auto p = r.getEndEffectorPose(q);
            return py::make_tuple(py::make_tuple(p.first.x(), p.first.y(), p.first.z()),
                                  py::make_tuple(p.second.w(), p.second.x(), p.second.y(), p.second.z()));
        }, py::arg("q"), "Tip pose ((x, y, z), (w, x, y, z))")
        .def("ik", [](Robot& r, const std::vector<double>& pos, const std::vector<double>& quat,
                      const std::vector<double>& guess) {
            bool converged = false;
            auto q = r.computeIK(Vector3(pos[0], pos[1], pos[2]),
                                 Quaternion(quat[0], quat[1], quat[2], quat[3]), guess, &converged);
            return py::make_tuple(q, converged);
        }, py::arg("position"), py::arg("quaternion"), py::arg("initial_guess") = std::vector<double>(),
           "(q, converged)")
        .def("move_j", [](Robot& r, const std::vector<double>& start, const std::vector<double>& goal,
                          size_t waypoints) {
            Trajectory t = r.moveJ(start, goal, waypoints);
            return py::make_tuple(t.getPositionMatrix(), t.success);
        }, py::arg("start"), py::arg("goal"), py::arg("waypoints") = 50,
           "Minimum-jerk joint-space trajectory: (plan [waypoints, dof], success)")
        .def("move_l", [](Robot& r, const std::vector<double>& start, const std::vector<double>& pos,
                          const std::vector<double>& quat, size_t waypoints) {
            Trajectory t = r.moveL(start, Vector3(pos[0], pos[1], pos[2]),
                                   Quaternion(quat[0], quat[1], quat[2], quat[3]), waypoints);
            return py::make_tuple(t.getPositionMatrix(), t.success);
        }, py::arg("start"), py::arg("goal_position"), py::arg("goal_quaternion"), py::arg("waypoints") = 50,
           "Straight-line Cartesian trajectory: (plan [waypoints, dof], success). success is False if IK "
           "failed somewhere along the line")
        .def_property_readonly("dt", [](Robot& r) { return r.getOptimizerConfig().dt; });
#endif
}
