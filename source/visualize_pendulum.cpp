// visualize_pendulum.cpp
// GPU-accelerated "particle pour" visualization.
// 2000 rigid body spheres pour into a box with gravity, showcasing
// the CUDA spatial hash broadphase and collision resolution at scale.
// Renders frames to ffmpeg pipe for GIF output.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#include "phys/phys.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Color { uint8_t r, g, b; };

static Color heat_color(float t) {
    // 0=cool blue, 1=hot red/white
    t = std::min(std::max(t, 0.0f), 1.0f);
    float r, g, b;
    if (t < 0.33f) {
        float s = t / 0.33f;
        r = 0.15f + 0.2f * s; g = 0.4f + 0.4f * s; b = 0.9f;
    } else if (t < 0.66f) {
        float s = (t - 0.33f) / 0.33f;
        r = 0.35f + 0.55f * s; g = 0.8f + 0.1f * s; b = 0.9f - 0.6f * s;
    } else {
        float s = (t - 0.66f) / 0.34f;
        r = 0.9f + 0.1f * s; g = 0.9f - 0.5f * s; b = 0.3f - 0.2f * s;
    }
    return {(uint8_t)(r * 255), (uint8_t)(g * 255), (uint8_t)(b * 255)};
}

static void draw_filled_circle(std::vector<uint8_t>& fb, int W, int H,
                                int cx, int cy, int radius, Color c) {
    int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; dy++) {
        int y = cy + dy;
        if (y < 0 || y >= H) continue;
        int dx_max = (int)sqrtf((float)(r2 - dy * dy));
        for (int dx = -dx_max; dx <= dx_max; dx++) {
            int x = cx + dx;
            if (x < 0 || x >= W) continue;
            float shade = 1.0f - 0.3f * sqrtf((float)(dx*dx+dy*dy)) / radius;
            int idx = (y * W + x) * 3;
            fb[idx]   = (uint8_t)(c.r * shade);
            fb[idx+1] = (uint8_t)(c.g * shade);
            fb[idx+2] = (uint8_t)(c.b * shade);
        }
    }
}

static void draw_rect(std::vector<uint8_t>& fb, int W, int H,
                       int x0, int y0, int x1, int y1, Color c) {
    for (int y = std::max(0, y0); y < std::min(H, y1); y++) {
        for (int x = std::max(0, x0); x < std::min(W, x1); x++) {
            int idx = (y * W + x) * 3;
            fb[idx] = c.r; fb[idx+1] = c.g; fb[idx+2] = c.b;
        }
    }
}

int main() {
    const int W = 480, H = 360;
    const int TOTAL_FRAMES = 160;
    const int SIM_SUBSTEPS = 6;
    const float DT = 0.0015f;
    const float GRAVITY = -15.0f;

    // Spawn parameters - add bodies in waves
    const int MAX_BODIES = 2000;
    const int SPAWN_PER_FRAME = 16;

    // 2D domain in the XY plane; z is shared by every body and vz stays 0
    float domain_x = 12.0f;
    float domain_y = 9.0f;  // 4:3 aspect
    float domain_z = 1.0f;
    float min_radius = 0.08f;
    float max_radius = 0.15f;

    srand(77);

    // All MAX_BODIES exist in the model from the start; bodies are switched on
    // as they spawn.
    phys::ModelDesc desc;
    desc.solver = phys::Solver::Particle;
    desc.gravity = {0.0f, GRAVITY, 0.0f};
    desc.bounds_lo = {0.0f, 0.0f, 0.0f};
    desc.bounds_hi = {domain_x, domain_y, domain_z};
    desc.wall_restitution = 0.5f;

    phys::HostState sys(MAX_BODIES);
    for (int i = 0; i < MAX_BODIES; i++) {
        float r = min_radius + ((float)rand()/RAND_MAX) * (max_radius - min_radius);
        phys::BodyDesc body = phys::BodyDesc::sphere(r, M_PI * r * r);
        body.restitution = 0.5f;
        desc.bodies.push_back(body);
        sys.enabled[i] = 0;
    }
    int active_bodies = 0;

    phys::World world(desc, 1, phys::Device::CUDA);

    std::vector<uint8_t> fb(W * H * 3);

    std::string cmd = "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 480x360 -r 30 "
                      "-i pipe:0 -vf \"fps=24,split[s0][s1];"
                      "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
                      "[s1][p]paletteuse=dither=floyd_steinberg\" "
                      "-loop 0 /tmp/particle_pour.gif 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) { std::cerr << "Failed to open ffmpeg\n"; return 1; }

    std::cout << "Rendering " << TOTAL_FRAMES << " frames, up to "
              << MAX_BODIES << " GPU-simulated bodies..." << std::endl;

    float scale_x = (float)W / domain_x;
    float scale_y = (float)H / domain_y;
    float max_speed = 10.0f;

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        // Spawn new bodies from the top
        if (active_bodies < MAX_BODIES) {
            int to_spawn = std::min(SPAWN_PER_FRAME, MAX_BODIES - active_bodies);
            for (int s = 0; s < to_spawn; s++) {
                int i = active_bodies++;
                // Spawn from two streams at top
                float stream_x;
                if (s % 2 == 0)
                    stream_x = domain_x * 0.3f + ((float)rand()/RAND_MAX - 0.5f) * 1.5f;
                else
                    stream_x = domain_x * 0.7f + ((float)rand()/RAND_MAX - 0.5f) * 1.5f;

                sys.px[i] = stream_x;
                sys.py[i] = domain_y - desc.bodies[i].size.x - 0.1f;
                sys.pz[i] = domain_z * 0.5f;
                sys.vx[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
                sys.vy[i] = -((float)rand()/RAND_MAX) * 3.0f;
                sys.vz[i] = 0.0f;
                sys.enabled[i] = 1;
            }
            world.set_state(sys);
        }

        // Run GPU simulation
        world.step(DT, SIM_SUBSTEPS);
        world.get_state(sys);

        // Clear framebuffer
        for (int p = 0; p < W * H; p++) {
            fb[p*3] = 10; fb[p*3+1] = 12; fb[p*3+2] = 20;
        }

        // Draw container walls
        Color wall_col = {45, 55, 70};
        draw_rect(fb, W, H, 0, 0, 3, H, wall_col);               // left
        draw_rect(fb, W, H, W-3, 0, W, H, wall_col);             // right
        draw_rect(fb, W, H, 0, H-3, W, H, wall_col);             // bottom
        draw_rect(fb, W, H, 0, 0, W, 3, wall_col);               // top

        // Draw bodies
        for (int i = 0; i < active_bodies; i++) {
            int cx = (int)(sys.px[i] * scale_x);
            int cy = H - (int)(sys.py[i] * scale_y);  // flip Y
            int r = std::max(2, (int)(desc.bodies[i].size.x * scale_x));

            float speed = sqrtf(sys.vx[i]*sys.vx[i] + sys.vy[i]*sys.vy[i]);
            Color col = heat_color(speed / max_speed);
            draw_filled_circle(fb, W, H, cx, cy, r, col);
        }

        fwrite(fb.data(), 1, W * H * 3, pipe);

        if (frame % 30 == 0)
            std::cout << "  Frame " << frame << "/" << TOTAL_FRAMES
                      << " (" << active_bodies << " bodies)" << std::endl;
    }

    pclose(pipe);
    std::cout << "Done! GIF saved to /tmp/particle_pour.gif" << std::endl;
    return 0;
}
