// visualize_rigid.cpp
// Rigid solver demo: a rain of spheres, capsules and boxes lands on a
// pyramid of boxes. Simulated with libphys and drawn with a small
// orthographic, flat-shaded software renderer into a GIF via an ffmpeg pipe.
//
//   ./visualize_rigid [output.gif] [cpu|cuda]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "phys/phys.h"

using namespace phys;

namespace {

const int W = 480, H = 360;

struct V { float x, y, z; };
V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V operator*(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V normalize(V a) { return a * (1.0f / std::sqrt(dot(a, a))); }

V rotate(const HostState& s, int g, V v) {
    float w = s.qw[g], x = s.qx[g], y = s.qy[g], z = s.qz[g];
    V u = {x, y, z};
    V t = {2 * (u.y * v.z - u.z * v.y), 2 * (u.z * v.x - u.x * v.z), 2 * (u.x * v.y - u.y * v.x)};
    V c = {u.y * t.z - u.z * t.y, u.z * t.x - u.x * t.z, u.x * t.y - u.y * t.x};
    return v + t * w + c;
}

struct Color { float r, g, b; };

// Orthographic camera looking down at the scene from the front-right.
struct Camera {
    V right, up, fwd;
    float scale = 52.0f, cx = W * 0.5f, cy = H * 0.62f;
    Camera() {
        float yaw = 0.6f, pitch = 0.42f;
        fwd = normalize({-std::sin(yaw) * std::cos(pitch), -std::sin(pitch),
                         -std::cos(yaw) * std::cos(pitch)});
        right = normalize({std::cos(yaw), 0.0f, -std::sin(yaw)});
        up = {right.y * fwd.z - right.z * fwd.y, right.z * fwd.x - right.x * fwd.z,
              right.x * fwd.y - right.y * fwd.x};
    }
    void project(V p, float& sx, float& sy) const {
        sx = cx + dot(p, right) * scale;
        sy = cy - dot(p, up) * scale;
    }
    float depth(V p) const { return dot(p, fwd); }
};

struct Canvas {
    std::vector<uint8_t> px = std::vector<uint8_t>(W * H * 3);
    void set(int x, int y, Color c) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        int i = (y * W + x) * 3;
        px[i] = (uint8_t)std::min(255.0f, c.r * 255);
        px[i + 1] = (uint8_t)std::min(255.0f, c.g * 255);
        px[i + 2] = (uint8_t)std::min(255.0f, c.b * 255);
    }
    void clear() {
        for (int y = 0; y < H; y++) {
            float t = (float)y / H;
            Color c = {0.07f + 0.05f * t, 0.08f + 0.06f * t, 0.12f + 0.06f * t};
            for (int x = 0; x < W; x++) set(x, y, c);
        }
    }
    // Convex polygon fill.
    void polygon(const float* xs, const float* ys, int n, Color c) {
        float x0 = W, x1 = 0, y0 = H, y1 = 0;
        for (int i = 0; i < n; i++) {
            x0 = std::min(x0, xs[i]); x1 = std::max(x1, xs[i]);
            y0 = std::min(y0, ys[i]); y1 = std::max(y1, ys[i]);
        }
        float area = 0;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            area += xs[i] * ys[j] - xs[j] * ys[i];
        }
        float sign = area >= 0 ? 1.0f : -1.0f;
        for (int y = std::max(0, (int)y0); y <= std::min(H - 1, (int)y1); y++) {
            for (int x = std::max(0, (int)x0); x <= std::min(W - 1, (int)x1); x++) {
                bool inside = true;
                for (int i = 0; i < n && inside; i++) {
                    int j = (i + 1) % n;
                    float e = (xs[j] - xs[i]) * (y + 0.5f - ys[i]) - (ys[j] - ys[i]) * (x + 0.5f - xs[i]);
                    inside = e * sign >= 0;
                }
                if (inside) set(x, y, c);
            }
        }
    }
    void disc(float cx, float cy, float r, Color c, bool shade) {
        for (int y = (int)(cy - r); y <= (int)(cy + r); y++) {
            for (int x = (int)(cx - r); x <= (int)(cx + r); x++) {
                float dx = x + 0.5f - cx, dy = y + 0.5f - cy;
                float d2 = dx * dx + dy * dy;
                if (d2 > r * r) continue;
                float k = 1.0f;
                if (shade) {  // light from the upper left
                    float lx = (dx + 0.4f * r) / r, ly = (dy + 0.4f * r) / r;
                    k = 1.05f - 0.45f * std::min(1.0f, std::sqrt(lx * lx + ly * ly));
                }
                set(x, y, {c.r * k, c.g * k, c.b * k});
            }
        }
    }
    void line(float x0, float y0, float x1, float y1, Color c) {
        int steps = (int)std::max(std::fabs(x1 - x0), std::fabs(y1 - y0)) + 1;
        for (int i = 0; i <= steps; i++) {
            float t = (float)i / steps;
            set((int)(x0 + (x1 - x0) * t), (int)(y0 + (y1 - y0) * t), c);
        }
    }
};

Color shade(Color c, V n) {
    V light = normalize({-0.4f, 1.0f, 0.6f});
    float k = 0.35f + 0.65f * std::max(0.0f, dot(n, light));
    return {c.r * k, c.g * k, c.b * k};
}

void draw_ground(Canvas& cv, const Camera& cam) {
    const float L = 6.0f;
    float xs[4], ys[4];
    V corners[4] = {{-L, 0, -L}, {L, 0, -L}, {L, 0, L}, {-L, 0, L}};
    for (int i = 0; i < 4; i++) cam.project(corners[i], xs[i], ys[i]);
    cv.polygon(xs, ys, 4, {0.16f, 0.18f, 0.22f});
    for (float t = -L; t <= L + 1e-3f; t += 1.0f) {
        float ax, ay, bx, by;
        cam.project({t, 0, -L}, ax, ay); cam.project({t, 0, L}, bx, by);
        cv.line(ax, ay, bx, by, {0.24f, 0.27f, 0.32f});
        cam.project({-L, 0, t}, ax, ay); cam.project({L, 0, t}, bx, by);
        cv.line(ax, ay, bx, by, {0.24f, 0.27f, 0.32f});
    }
}

void draw_body(Canvas& cv, const Camera& cam, const BodyDesc& body, const HostState& s, int g,
               Color c) {
    V x = {s.px[g], s.py[g], s.pz[g]};
    float sx, sy;
    if (body.shape == Shape::Sphere) {
        cam.project(x, sx, sy);
        cv.disc(sx, sy, body.size.x * cam.scale, c, true);
    } else if (body.shape == Shape::Capsule) {
        V axis = rotate(s, g, {0, body.size.y, 0});
        float ax, ay, bx, by;
        cam.project(x - axis, ax, ay);
        cam.project(x + axis, bx, by);
        float r = body.size.x * cam.scale;
        float dx = bx - ax, dy = by - ay, len = std::sqrt(dx * dx + dy * dy) + 1e-6f;
        float nx = -dy / len * r, ny = dx / len * r;
        float qx[4] = {ax + nx, bx + nx, bx - nx, ax - nx}, qy[4] = {ay + ny, by + ny, by - ny, ay - ny};
        Color body_c = {c.r * 0.85f, c.g * 0.85f, c.b * 0.85f};
        // Far end first so the near cap overlaps the body.
        bool a_far = cam.depth(x - axis) > cam.depth(x + axis);
        cv.disc(a_far ? ax : bx, a_far ? ay : by, r, body_c, true);
        cv.polygon(qx, qy, 4, body_c);
        cv.disc(a_far ? bx : ax, a_far ? by : ay, r, c, true);
    } else if (body.shape == Shape::Box) {
        V h = {body.size.x, body.size.y, body.size.z};
        for (int axis = 0; axis < 3; axis++) {
            for (float sign : {-1.0f, 1.0f}) {
                V n_local = {0, 0, 0};
                (axis == 0 ? n_local.x : axis == 1 ? n_local.y : n_local.z) = sign;
                V n = rotate(s, g, n_local);
                if (dot(n, cam.fwd) >= 0) continue;  // back face
                float xs[4], ys[4];
                for (int k = 0; k < 4; k++) {
                    float u = (k == 1 || k == 2) ? 1.0f : -1.0f, v = (k >= 2) ? 1.0f : -1.0f;
                    V local;
                    if (axis == 0) local = {sign * h.x, u * h.y, v * h.z};
                    else if (axis == 1) local = {u * h.x, sign * h.y, v * h.z};
                    else local = {u * h.x, v * h.y, sign * h.z};
                    cam.project(x + rotate(s, g, local), xs[k], ys[k]);
                }
                cv.polygon(xs, ys, 4, shade(c, n));
                Color edge = {c.r * 0.3f, c.g * 0.3f, c.b * 0.3f};
                for (int k = 0; k < 4; k++)
                    cv.line(xs[k], ys[k], xs[(k + 1) % 4], ys[(k + 1) % 4], edge);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = argc > 1 ? argv[1] : "/tmp/rigid_pile.gif";
    const int FRAMES = 240, PYRAMID = 4, RAIN = 36;

    ModelDesc desc;
    desc.substeps = 20;
    desc.bodies.push_back(BodyDesc::plane());
    std::vector<Color> colors = {{0, 0, 0}};

    // A pyramid of boxes.
    std::vector<V> start;
    start.push_back({0, 0, 0});
    for (int row = 0; row < PYRAMID; row++) {
        for (int i = 0; i < PYRAMID - row; i++) {
            BodyDesc b = BodyDesc::box({0.45f, 0.3f, 0.45f}, 1.0f);
            b.restitution = 0.1f;
            b.friction = 0.6f;
            desc.bodies.push_back(b);
            colors.push_back({0.93f, 0.62f - 0.08f * row, 0.25f});
            start.push_back({(i - (PYRAMID - row - 1) * 0.5f) * 0.95f, 0.3f + row * 0.6f, 0.0f});
        }
    }
    // Shapes that rain down, staggered so they arrive over time.
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    for (int i = 0; i < RAIN; i++) {
        BodyDesc b;
        Color c;
        switch (i % 3) {
            case 0: b = BodyDesc::sphere(0.22f + 0.1f * unit(rng), 0.8f); c = {0.30f, 0.72f, 0.95f}; break;
            case 1: b = BodyDesc::capsule(0.13f, 0.3f, 0.8f); c = {0.55f, 0.85f, 0.45f}; break;
            default: b = BodyDesc::box({0.2f, 0.2f, 0.2f}, 0.8f); c = {0.88f, 0.42f, 0.62f}; break;
        }
        b.restitution = 0.3f;
        desc.bodies.push_back(b);
        colors.push_back(c);
        start.push_back({(unit(rng) - 0.5f) * 3.0f, 4.0f + i * 0.45f, (unit(rng) - 0.5f) * 1.6f});
    }

    int nbody = (int)desc.bodies.size();
    HostState s(nbody);
    for (int g = 1; g < nbody; g++) {
        s.px[g] = start[g].x; s.py[g] = start[g].y; s.pz[g] = start[g].z;
        if (g > PYRAMID * (PYRAMID + 1) / 2) {  // tumbling rain
            float a = 3.0f * unit(rng);
            V ax = normalize({unit(rng) - 0.5f, unit(rng) - 0.5f, unit(rng) - 0.5f});
            s.qw[g] = std::cos(a / 2);
            s.qx[g] = ax.x * std::sin(a / 2); s.qy[g] = ax.y * std::sin(a / 2); s.qz[g] = ax.z * std::sin(a / 2);
            s.vx[g] = (unit(rng) - 0.5f) * 1.0f;
        }
    }

    // A single scene with many bodies: the CPU backend is the faster choice
    // here. The CUDA backend solves each env's contacts on one GPU thread and
    // pays off when stepping many envs at once (see bench_envs).
    World world(desc, 1, argc > 2 && std::string(argv[2]) == "cuda" ? Device::CUDA : Device::CPU);
    world.set_state(s);

    std::string cmd = "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 480x360 -r 30 -i pipe:0 "
                      "-vf \"split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];"
                      "[s1][p]paletteuse=dither=floyd_steinberg\" -loop 0 " + out + " 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) { std::cerr << "Failed to open ffmpeg\n"; return 1; }

    Camera cam;
    Canvas cv;
    std::vector<int> order(nbody - 1);
    for (int frame = 0; frame < FRAMES; frame++) {
        world.step(1.0f / 60.0f, 2);  // 30 fps playback of 60 Hz physics
        world.get_state(s);

        cv.clear();
        draw_ground(cv, cam);
        for (int i = 0; i < nbody - 1; i++) order[i] = i + 1;
        std::sort(order.begin(), order.end(), [&](int a, int b) {  // painter's algorithm
            return cam.depth({s.px[a], s.py[a], s.pz[a]}) > cam.depth({s.px[b], s.py[b], s.pz[b]});
        });
        for (int g : order) draw_body(cv, cam, desc.bodies[g], s, g, colors[g]);
        fwrite(cv.px.data(), 1, cv.px.size(), pipe);
    }
    pclose(pipe);
    std::cout << "Wrote " << out << " (" << nbody - 1 << " bodies, " << FRAMES << " frames)\n";
    return 0;
}
