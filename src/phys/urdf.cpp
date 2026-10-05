#include "phys/urdf.h"

#include <tinyxml2.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <set>
#include <stdexcept>

namespace phys {

namespace {

// --- pose algebra (double precision) -------------------------------------------

void qmul(const double a[4], const double b[4], double out[4]) {
    double w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    double x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    double y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    double z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
    out[0] = w; out[1] = x; out[2] = y; out[3] = z;
}

void qrot(const double q[4], const double v[3], double out[3]) {
    double t[3] = {2 * (q[2] * v[2] - q[3] * v[1]), 2 * (q[3] * v[0] - q[1] * v[2]),
                   2 * (q[1] * v[1] - q[2] * v[0])};
    out[0] = v[0] + q[0] * t[0] + (q[2] * t[2] - q[3] * t[1]);
    out[1] = v[1] + q[0] * t[1] + (q[3] * t[0] - q[1] * t[2]);
    out[2] = v[2] + q[0] * t[2] + (q[1] * t[1] - q[2] * t[0]);
}

Pose compose(const Pose& a, const Pose& b) {
    Pose c;
    double r[3];
    qrot(a.q, b.p, r);
    for (int i = 0; i < 3; i++) c.p[i] = a.p[i] + r[i];
    qmul(a.q, b.q, c.q);
    return c;
}

Pose inverse(const Pose& a) {
    Pose c;
    c.q[0] = a.q[0]; c.q[1] = -a.q[1]; c.q[2] = -a.q[2]; c.q[3] = -a.q[3];
    double r[3];
    qrot(c.q, a.p, r);
    for (int i = 0; i < 3; i++) c.p[i] = -r[i];
    return c;
}

Pose from_axis_angle(const double axis[3], double angle) {
    Pose p;
    double s = std::sin(angle / 2);
    p.q[0] = std::cos(angle / 2);
    p.q[1] = axis[0] * s; p.q[2] = axis[1] * s; p.q[3] = axis[2] * s;
    return p;
}

// URDF rpy: fixed-axis roll (x), pitch (y), yaw (z): R = Rz Ry Rx.
Pose from_xyz_rpy(const double xyz[3], const double rpy[3]) {
    double x[3] = {1, 0, 0}, y[3] = {0, 1, 0}, z[3] = {0, 0, 1};
    Pose rz = from_axis_angle(z, rpy[2]), ry = from_axis_angle(y, rpy[1]), rx = from_axis_angle(x, rpy[0]);
    Pose p = compose(compose(rz, ry), rx);
    for (int i = 0; i < 3; i++) p.p[i] = xyz[i];
    return p;
}

// Rotation taking the x axis onto unit vector a.
Pose x_onto(const double a[3]) {
    Pose p;
    if (a[0] < -0.999999) {
        p.q[0] = 0; p.q[1] = 0; p.q[2] = 1; p.q[3] = 0;
        return p;
    }
    double w = 1 + a[0], qy = -a[2], qz = a[1];
    double n = std::sqrt(w * w + qy * qy + qz * qz);
    p.q[0] = w / n; p.q[1] = 0; p.q[2] = qy / n; p.q[3] = qz / n;
    return p;
}

// Symmetric 3x3 eigen-decomposition (Jacobi): returns eigenvalues and the
// rotation whose columns are the eigenvectors, as a quaternion.
void principal_axes(double I[3][3], double moments[3], double q[4]) {
    double V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double A[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A[i][j] = I[i][j];
    for (int sweep = 0; sweep < 50; sweep++) {
        double off = std::fabs(A[0][1]) + std::fabs(A[0][2]) + std::fabs(A[1][2]);
        if (off < 1e-15) break;
        for (int p = 0; p < 2; p++) {
            for (int r = p + 1; r < 3; r++) {
                if (std::fabs(A[p][r]) < 1e-18) continue;
                double theta = 0.5 * std::atan2(2 * A[p][r], A[r][r] - A[p][p]);
                double c = std::cos(theta), s = std::sin(theta);
                for (int k = 0; k < 3; k++) {  // A = J^T A J
                    double akp = A[k][p], akr = A[k][r];
                    A[k][p] = c * akp - s * akr;
                    A[k][r] = s * akp + c * akr;
                }
                for (int k = 0; k < 3; k++) {
                    double apk = A[p][k], ark = A[r][k];
                    A[p][k] = c * apk - s * ark;
                    A[r][k] = s * apk + c * ark;
                }
                for (int k = 0; k < 3; k++) {
                    double vkp = V[k][p], vkr = V[k][r];
                    V[k][p] = c * vkp - s * vkr;
                    V[k][r] = s * vkp + c * vkr;
                }
            }
        }
    }
    for (int i = 0; i < 3; i++) moments[i] = A[i][i];
    // Make V a proper rotation.
    double det = V[0][0] * (V[1][1] * V[2][2] - V[1][2] * V[2][1]) -
                 V[0][1] * (V[1][0] * V[2][2] - V[1][2] * V[2][0]) +
                 V[0][2] * (V[1][0] * V[2][1] - V[1][1] * V[2][0]);
    if (det < 0)
        for (int k = 0; k < 3; k++) V[k][2] = -V[k][2];
    double tr = V[0][0] + V[1][1] + V[2][2];
    if (tr > 0) {
        double s = 2 * std::sqrt(tr + 1);
        q[0] = 0.25 * s;
        q[1] = (V[2][1] - V[1][2]) / s;
        q[2] = (V[0][2] - V[2][0]) / s;
        q[3] = (V[1][0] - V[0][1]) / s;
    } else if (V[0][0] > V[1][1] && V[0][0] > V[2][2]) {
        double s = 2 * std::sqrt(1 + V[0][0] - V[1][1] - V[2][2]);
        q[0] = (V[2][1] - V[1][2]) / s; q[1] = 0.25 * s;
        q[2] = (V[0][1] + V[1][0]) / s; q[3] = (V[0][2] + V[2][0]) / s;
    } else if (V[1][1] > V[2][2]) {
        double s = 2 * std::sqrt(1 + V[1][1] - V[0][0] - V[2][2]);
        q[0] = (V[0][2] - V[2][0]) / s; q[1] = (V[0][1] + V[1][0]) / s;
        q[2] = 0.25 * s; q[3] = (V[1][2] + V[2][1]) / s;
    } else {
        double s = 2 * std::sqrt(1 + V[2][2] - V[0][0] - V[1][1]);
        q[0] = (V[1][0] - V[0][1]) / s; q[1] = (V[0][2] + V[2][0]) / s;
        q[2] = (V[1][2] + V[2][1]) / s; q[3] = 0.25 * s;
    }
}

void read3(const tinyxml2::XMLElement* e, const char* attr, double out[3]) {
    const char* s = e ? e->Attribute(attr) : nullptr;
    if (s) std::sscanf(s, "%lf %lf %lf", &out[0], &out[1], &out[2]);
}

Pose read_origin(const tinyxml2::XMLElement* parent) {
    double xyz[3] = {0, 0, 0}, rpy[3] = {0, 0, 0};
    const tinyxml2::XMLElement* o = parent ? parent->FirstChildElement("origin") : nullptr;
    read3(o, "xyz", xyz);
    read3(o, "rpy", rpy);
    return from_xyz_rpy(xyz, rpy);
}

Vec3 v3(const double p[3]) { return {(float)p[0], (float)p[1], (float)p[2]}; }

Quat q4(const double q[4]) {
    Quat r;
    r.w = (float)q[0]; r.x = (float)q[1]; r.y = (float)q[2]; r.z = (float)q[3];
    return r;
}

}  // namespace

UrdfModel load_urdf(const std::string& path, const UrdfOptions& opt) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("load_urdf: cannot read " + path);
    const tinyxml2::XMLElement* robot = doc.FirstChildElement("robot");
    if (!robot) throw std::runtime_error("load_urdf: no <robot> in " + path);

    UrdfModel m;
    std::map<std::string, double> mass;
    std::map<std::string, Vec3> moments;
    std::vector<std::string> link_order, joint_order;
    std::map<std::string, std::vector<std::string>> children;  // link -> child joints (file order)
    std::map<std::string, std::vector<double>> limit;          // joint -> lower, upper, effort

    for (auto* l = robot->FirstChildElement("link"); l; l = l->NextSiblingElement("link")) {
        std::string name = l->Attribute("name");
        UrdfModel::LinkInfo info;
        if (auto* in = l->FirstChildElement("inertial")) {
            double mval = 0;
            if (auto* me = in->FirstChildElement("mass")) mval = me->DoubleAttribute("value");
            if (mval > 0) {
                info.has_mass = true;
                double I[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                if (auto* ie = in->FirstChildElement("inertia")) {
                    I[0][0] = ie->DoubleAttribute("ixx"); I[1][1] = ie->DoubleAttribute("iyy");
                    I[2][2] = ie->DoubleAttribute("izz");
                    I[0][1] = I[1][0] = ie->DoubleAttribute("ixy");
                    I[0][2] = I[2][0] = ie->DoubleAttribute("ixz");
                    I[1][2] = I[2][1] = ie->DoubleAttribute("iyz");
                }
                double mom[3];
                Pose principal;
                principal_axes(I, mom, principal.q);
                info.inertial = compose(read_origin(in), principal);
                mass[name] = mval;
                // Guard against degenerate inertias in the file.
                double floor = 1e-6 * std::fmax(std::fmax(mom[0], mom[1]), std::fmax(mom[2], 1e-9));
                moments[name] = {(float)std::fmax(mom[0], floor), (float)std::fmax(mom[1], floor),
                                 (float)std::fmax(mom[2], floor)};
            }
        }
        m.links[name] = info;
        link_order.push_back(name);
    }

    std::set<std::string> child_links;
    for (auto* j = robot->FirstChildElement("joint"); j; j = j->NextSiblingElement("joint")) {
        UrdfModel::JointInfo info;
        info.name = j->Attribute("name");
        info.type = j->Attribute("type");
        info.parent = j->FirstChildElement("parent")->Attribute("link");
        info.child = j->FirstChildElement("child")->Attribute("link");
        info.origin = read_origin(j);
        read3(j->FirstChildElement("axis"), "xyz", info.axis);
        double n = std::sqrt(info.axis[0] * info.axis[0] + info.axis[1] * info.axis[1] +
                             info.axis[2] * info.axis[2]);
        if (n == 0) throw std::runtime_error("load_urdf: zero axis on joint " + info.name);
        for (double& a : info.axis) a /= n;
        if (auto* lim = j->FirstChildElement("limit"))
            limit[info.name] = {lim->DoubleAttribute("lower"), lim->DoubleAttribute("upper"),
                                lim->DoubleAttribute("effort")};
        if (!m.links.count(info.parent) || !m.links.count(info.child))
            throw std::runtime_error("load_urdf: joint " + info.name + " references an unknown link");
        if (!child_links.insert(info.child).second)
            throw std::runtime_error("load_urdf: link " + info.child + " has two parents");
        m.links[info.child].parent_joint = info.name;
        children[info.parent].push_back(info.name);
        m.joints[info.name] = info;
        joint_order.push_back(info.name);
    }
    for (const std::string& l : link_order) {
        if (!child_links.count(l)) {
            if (!m.root.empty()) throw std::runtime_error("load_urdf: more than one root link");
            m.root = l;
        }
    }
    if (m.root.empty()) throw std::runtime_error("load_urdf: no root link");

    m.base.p[0] = opt.base_position.x; m.base.p[1] = opt.base_position.y;
    m.base.p[2] = opt.base_position.z;
    m.base.q[0] = opt.base_rotation.w; m.base.q[1] = opt.base_rotation.x;
    m.base.q[2] = opt.base_rotation.y; m.base.q[3] = opt.base_rotation.z;

    m.model.gravity = opt.gravity;
    // Every link is held by a body (or the world, -1) through a fixed
    // transform: frame of that body -> frame of the link.
    std::map<std::string, std::pair<int, Pose>> held;
    held[m.root] = {-1, m.base};

    std::function<void(const std::string&)> visit = [&](const std::string& link) {
        for (const std::string& jn : children[link]) {
            UrdfModel::JointInfo& j = m.joints[jn];
            UrdfModel::LinkInfo& child = m.links[j.child];
            const auto& parent = held[link];
            Pose joint_in_parent_body = compose(parent.second, j.origin);
            bool movable = j.type == "revolute" || j.type == "continuous" || j.type == "prismatic";
            if (!movable && j.type != "fixed")
                throw std::runtime_error("load_urdf: unsupported joint type " + j.type + " (" + jn + ")");

            if (!movable && (parent.first < 0 || !child.has_mass)) {
                // Welded to the world, or a massless frame: no body of its own.
                held[j.child] = {parent.first, joint_in_parent_body};
            } else {
                if (!child.has_mass)
                    throw std::runtime_error("load_urdf: movable joint " + jn + " drives massless link " +
                                             j.child);
                int body = (int)m.model.bodies.size();
                BodyDesc bd = BodyDesc::none((float)mass[j.child], moments[j.child]);
                if (opt.sphere_radius > 0) {
                    bd.shape = Shape::Sphere;
                    bd.size = {opt.sphere_radius, 0, 0};
                }
                m.model.bodies.push_back(bd);
                m.body_links.push_back(j.child);
                child.body = body;
                held[j.child] = {body, inverse(child.inertial)};

                Pose frame = movable ? x_onto(j.axis) : Pose();
                Pose pf = compose(joint_in_parent_body, frame);
                Pose cf = compose(inverse(child.inertial), frame);
                JointDesc jd;
                jd.type = !movable ? JointType::Fixed
                                   : (j.type == "prismatic" ? JointType::Slider : JointType::Hinge);
                jd.parent = parent.first;
                jd.child = body;
                jd.parent_anchor = v3(pf.p);
                jd.parent_frame = q4(pf.q);
                jd.child_anchor = v3(cf.p);
                jd.child_frame = q4(cf.q);
                if (movable) {
                    j.dof = (int)m.joint_names.size();
                    m.joint_names.push_back(jn);
                    double lo = -1e30, hi = 1e30, effort = 0;
                    if (limit.count(jn)) {
                        effort = limit[jn][2];
                        if (j.type != "continuous") {
                            lo = limit[jn][0];
                            hi = limit[jn][1];
                            if (hi > lo) {
                                jd.limited = true;
                                jd.lower = (float)lo;
                                jd.upper = (float)hi;
                            }
                        }
                    }
                    m.lower.push_back((float)lo);
                    m.upper.push_back((float)hi);
                    jd.actuator = opt.actuator;
                    jd.kp = opt.kp;
                    jd.kd = opt.kd;
                    jd.damping = opt.joint_damping;
                    if (opt.effort_limits && effort > 0) jd.max_force = (float)effort;
                }
                m.model.joints.push_back(jd);
            }
            visit(j.child);
        }
    };
    visit(m.root);
    if (m.model.bodies.empty()) throw std::runtime_error("load_urdf: no movable bodies in " + path);
    return m;
}

Pose UrdfModel::link_pose(const std::string& link, const std::vector<double>& q) const {
    auto it = links.find(link);
    if (it == links.end()) throw std::invalid_argument("link_pose: unknown link " + link);
    if (it->second.parent_joint.empty()) return base;
    const JointInfo& j = joints.at(it->second.parent_joint);
    Pose t = compose(link_pose(j.parent, q), j.origin);
    if (j.dof >= 0) {
        if ((int)q.size() <= j.dof) throw std::invalid_argument("link_pose: too few joint positions");
        if (j.type == "prismatic") {
            Pose s;
            for (int i = 0; i < 3; i++) s.p[i] = j.axis[i] * q[j.dof];
            t = compose(t, s);
        } else {
            t = compose(t, from_axis_angle(j.axis, q[j.dof]));
        }
    }
    return t;
}

void UrdfModel::set_configuration(HostState& s, int env, const std::vector<double>& q) const {
    int nbody = (int)model.bodies.size();
    if (s.size() < (env + 1) * nbody) throw std::invalid_argument("set_configuration: state too small");
    for (int b = 0; b < nbody; b++) {
        Pose p = compose(link_pose(body_links[b], q), links.at(body_links[b]).inertial);
        int g = env * nbody + b;
        s.px[g] = (float)p.p[0]; s.py[g] = (float)p.p[1]; s.pz[g] = (float)p.p[2];
        s.qw[g] = (float)p.q[0]; s.qx[g] = (float)p.q[1]; s.qy[g] = (float)p.q[2]; s.qz[g] = (float)p.q[3];
        s.vx[g] = s.vy[g] = s.vz[g] = s.wx[g] = s.wy[g] = s.wz[g] = 0.0f;
        s.enabled[g] = 1;
    }
}

}  // namespace phys
