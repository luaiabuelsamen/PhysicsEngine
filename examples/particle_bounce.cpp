// visualize.cpp
// Renders a 2D top-down view of the rigid body simulation as PPM frames,
// then pipes to ffmpeg to produce an animated GIF.

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include "phys/phys.h"

struct Color { uint8_t r, g, b; };

static Color velocity_color(float speed, float max_speed) {
    float t = std::min(speed / max_speed, 1.0f);

    // Vivid: teal -> green -> yellow -> orange -> red
    float r, g, b;
    if (t < 0.2f) {
        float s = t / 0.2f;
        r = 0.1f + 0.1f * s; g = 0.6f + 0.2f * s; b = 0.8f;
    } else if (t < 0.4f) {
        float s = (t - 0.2f) / 0.2f;
        r = 0.2f + 0.1f * s; g = 0.8f + 0.1f * s; b = 0.8f - 0.5f * s;
    } else if (t < 0.6f) {
        float s = (t - 0.4f) / 0.2f;
        r = 0.3f + 0.5f * s; g = 0.9f; b = 0.3f - 0.2f * s;
    } else if (t < 0.8f) {
        float s = (t - 0.6f) / 0.2f;
        r = 0.8f + 0.15f * s; g = 0.9f - 0.3f * s; b = 0.1f;
    } else {
        float s = (t - 0.8f) / 0.2f;
        r = 0.95f + 0.05f * s; g = 0.6f - 0.3f * s; b = 0.1f + 0.1f * s;
    }

    return {(uint8_t)(r * 255), (uint8_t)(g * 255), (uint8_t)(b * 255)};
}

static void draw_filled_circle(std::vector<uint8_t>& fb, int W, int H,
                                 int cx, int cy, int radius, Color col) {
    int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; dy++) {
        int y = cy + dy;
        if (y < 0 || y >= H) continue;
        int dx_max = (int)sqrtf((float)(r2 - dy * dy));
        for (int dx = -dx_max; dx <= dx_max; dx++) {
            int x = cx + dx;
            if (x < 0 || x >= W) continue;
            int idx = (y * W + x) * 3;

            // Slight shading for 3D effect
            float dist = sqrtf((float)(dx * dx + dy * dy));
            float shade = 1.0f - 0.3f * (dist / radius);

            // Additive blend for glow
            int nr = std::min(255, (int)(col.r * shade));
            int ng = std::min(255, (int)(col.g * shade));
            int nb = std::min(255, (int)(col.b * shade));

            fb[idx]     = (uint8_t)nr;
            fb[idx + 1] = (uint8_t)ng;
            fb[idx + 2] = (uint8_t)nb;
        }
    }
}

static void draw_border(std::vector<uint8_t>& fb, int W, int H, int thickness) {
    Color border = {60, 70, 90};
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (x < thickness || x >= W - thickness || y < thickness || y >= H - thickness) {
                int idx = (y * W + x) * 3;
                fb[idx] = border.r; fb[idx+1] = border.g; fb[idx+2] = border.b;
            }
        }
    }
}

int main() {
    const int W = 480, H = 360;
    const int NUM_BODIES = 200;
    const int TOTAL_FRAMES = 180;
    const int SIM_SUBSTEPS = 8;      // substeps per frame for smooth physics
    const float DT = 0.002f;
    const float GRAVITY = -9.81f;

    // 2D simulation: use XY plane, ignore Z
    float domain_x = 20.0f;
    float domain_y = 15.0f;  // matches aspect ratio
    float min_radius = 0.2f;
    float max_radius = 0.45f;

    srand(123);

    // 2D simulation in the XY plane: every body shares one z and has vz = 0,
    // so contact normals have no z component and the bodies stay planar.
    phys::ModelDesc desc;
    desc.solver = phys::Solver::Particle;
    desc.gravity = {0.0f, GRAVITY, 0.0f};
    desc.bounds_lo = {0.0f, 0.0f, 0.0f};
    desc.bounds_hi = {domain_x, domain_y, domain_y};
    desc.wall_restitution = 0.9f;

    phys::HostState sys(NUM_BODIES);
    for (int i = 0; i < NUM_BODIES; i++) {
        float r = min_radius + ((float)rand() / RAND_MAX) * (max_radius - min_radius);
        phys::BodyDesc body = phys::BodyDesc::sphere(r, M_PI * r * r);  // 2D mass ~ area
        body.restitution = 0.85f;
        desc.bodies.push_back(body);

        sys.px[i] = r + ((float)rand() / RAND_MAX) * (domain_x - 2 * r);
        sys.py[i] = r + ((float)rand() / RAND_MAX) * (domain_y - 2 * r);
        sys.pz[i] = domain_y * 0.5f;

        float speed = 6.0f;
        sys.vx[i] = (((float)rand() / RAND_MAX) - 0.5f) * speed;
        sys.vy[i] = (((float)rand() / RAND_MAX) - 0.5f) * speed + 2.0f;
    }

    phys::World world(desc, 1, phys::Device::CUDA);
    world.set_state(sys);

    std::vector<uint8_t> fb(W * H * 3);

    // Open pipe to ffmpeg -> GIF
    std::string cmd = "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 480x360 -r 30 "
                      "-i pipe:0 -vf \"fps=20,scale=480:360:flags=lanczos,"
                      "split[s0][s1];[s0]palettegen=max_colors=96:stats_mode=diff[p];"
                      "[s1][p]paletteuse=dither=floyd_steinberg\" "
                      "-loop 0 /tmp/particle_bounce.gif 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) {
        std::cerr << "Failed to open ffmpeg pipe\n";
        return 1;
    }

    float max_speed = 8.0f;

    std::cout << "Rendering " << TOTAL_FRAMES << " frames with " << NUM_BODIES << " bodies..." << std::endl;

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        world.step(DT, SIM_SUBSTEPS);
        world.get_state(sys);

        // Clear framebuffer (dark background)
        memset(fb.data(), 0x0E, W * H * 3);
        // Slightly different bg channels for visual warmth
        for (int p = 0; p < W * H; p++) {
            fb[p * 3]     = 12;  // R
            fb[p * 3 + 1] = 14;  // G
            fb[p * 3 + 2] = 22;  // B
        }

        draw_border(fb, W, H, 3);

        // Draw bodies
        float scale_x = (float)(W - 6) / domain_x;
        float scale_y = (float)(H - 6) / domain_y;

        for (int i = 0; i < NUM_BODIES; i++) {
            int cx = 3 + (int)(sys.px[i] * scale_x);
            int cy = H - 3 - (int)(sys.py[i] * scale_y);  // flip Y
            int r = std::max(2, (int)(desc.bodies[i].size.x * scale_x));

            float speed = sqrtf(sys.vx[i]*sys.vx[i] + sys.vy[i]*sys.vy[i]);
            Color col = velocity_color(speed, max_speed);
            draw_filled_circle(fb, W, H, cx, cy, r, col);
        }

        fwrite(fb.data(), 1, W * H * 3, pipe);

        if (frame % 50 == 0) {
            std::cout << "  Frame " << frame << "/" << TOTAL_FRAMES << std::endl;
        }
    }

    pclose(pipe);
    std::cout << "Done! GIF saved to /tmp/particle_bounce.gif" << std::endl;
    return 0;
}
