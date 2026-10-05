// visualize_chaos.cpp
// GPU-accelerated double pendulum chaos visualization.
// 500 pendulums with slightly different initial conditions integrated
// in parallel on CUDA, showing chaotic divergence with colored trails.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Color { uint8_t r, g, b; };

// Declared in cuda_pendulum.cu
extern "C" void cuda_integrate_pendulums(
    float* theta1, float* theta2,
    float* omega1, float* omega2,
    float* tip_x, float* tip_y,
    int num_pendulums, int substeps, float dt,
    float g, float L1, float L2, float m1, float m2);

static Color hsv_to_rgb(float h, float s, float v) {
    // h in [0,360), s,v in [0,1]
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r, g, b;
    if (h < 60)       { r=c; g=x; b=0; }
    else if (h < 120) { r=x; g=c; b=0; }
    else if (h < 180) { r=0; g=c; b=x; }
    else if (h < 240) { r=0; g=x; b=c; }
    else if (h < 300) { r=x; g=0; b=c; }
    else               { r=c; g=0; b=x; }
    return {(uint8_t)((r+m)*255), (uint8_t)((g+m)*255), (uint8_t)((b+m)*255)};
}

int main() {
    const int W = 480, H = 360;
    const int NUM_PENDULUMS = 500;
    const int TOTAL_FRAMES = 200;
    const int SUBSTEPS = 15;
    const float DT = 0.004f;

    float g = 9.81f, L1 = 1.0f, L2 = 1.0f, m1 = 1.0f, m2 = 1.0f;

    // Allocate state arrays
    std::vector<float> theta1(NUM_PENDULUMS), theta2(NUM_PENDULUMS);
    std::vector<float> omega1(NUM_PENDULUMS), omega2(NUM_PENDULUMS);
    std::vector<float> tip_x(NUM_PENDULUMS), tip_y(NUM_PENDULUMS);

    // Initialize with tiny spread around same starting angle
    float base_t1 = (float)(M_PI * 0.7);
    float base_t2 = (float)(M_PI * 0.5);
    float spread = 0.02f;  // total spread in radians

    for (int i = 0; i < NUM_PENDULUMS; i++) {
        float frac = (float)i / (NUM_PENDULUMS - 1);  // 0 to 1
        theta1[i] = base_t1 + (frac - 0.5f) * spread;
        theta2[i] = base_t2;
        omega1[i] = 0; omega2[i] = 0;
    }

    // Colors: rainbow based on initial condition index
    std::vector<Color> colors(NUM_PENDULUMS);
    for (int i = 0; i < NUM_PENDULUMS; i++) {
        float hue = 360.0f * (float)i / NUM_PENDULUMS;
        colors[i] = hsv_to_rgb(hue, 0.9f, 0.95f);
    }

    // Pivot and scale
    int cx = W / 2, cy = H / 4;
    float max_extent = L1 + L2;
    float avail = std::min((float)(H - cy - 10), (float)(cx - 10));
    float scale = avail * 0.75f / max_extent;

    std::vector<uint8_t> fb(W * H * 3);

    // Trail: accumulate into a persistent glow buffer
    std::vector<float> glow_r(W * H, 0), glow_g(W * H, 0), glow_b(W * H, 0);

    std::string cmd = "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 480x360 -r 30 "
                      "-i pipe:0 -vf \"fps=24,split[s0][s1];"
                      "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
                      "[s1][p]paletteuse=dither=floyd_steinberg\" "
                      "-loop 0 /tmp/pendulum_chaos.gif 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) { std::cerr << "Failed to open ffmpeg\n"; return 1; }

    std::cout << "Rendering " << TOTAL_FRAMES << " frames, "
              << NUM_PENDULUMS << " CUDA-parallel pendulums..." << std::endl;

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        // GPU-accelerated parallel integration of all pendulums
        cuda_integrate_pendulums(
            theta1.data(), theta2.data(),
            omega1.data(), omega2.data(),
            tip_x.data(), tip_y.data(),
            NUM_PENDULUMS, SUBSTEPS, DT,
            g, L1, L2, m1, m2);

        // Fade glow buffer slightly each frame for trail effect
        for (int p = 0; p < W * H; p++) {
            glow_r[p] *= 0.97f;
            glow_g[p] *= 0.97f;
            glow_b[p] *= 0.97f;
        }

        // Add current tip positions to glow
        for (int i = 0; i < NUM_PENDULUMS; i++) {
            int px = cx + (int)(tip_x[i] * scale);
            int py = cy + (int)(tip_y[i] * scale);
            // Draw a 2px dot
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int x = px + dx, y = py + dy;
                    if (x >= 0 && x < W && y >= 0 && y < H) {
                        int idx = y * W + x;
                        glow_r[idx] = std::min(255.0f, glow_r[idx] + colors[i].r * 0.15f);
                        glow_g[idx] = std::min(255.0f, glow_g[idx] + colors[i].g * 0.15f);
                        glow_b[idx] = std::min(255.0f, glow_b[idx] + colors[i].b * 0.15f);
                    }
                }
            }
        }

        // Render frame: dark bg + glow
        for (int p = 0; p < W * H; p++) {
            fb[p*3]   = (uint8_t)std::min(255.0f, 8.0f + glow_r[p]);
            fb[p*3+1] = (uint8_t)std::min(255.0f, 10.0f + glow_g[p]);
            fb[p*3+2] = (uint8_t)std::min(255.0f, 16.0f + glow_b[p]);
        }

        // Draw a few representative pendulum arms (every 100th)
        for (int i = 0; i < NUM_PENDULUMS; i += 100) {
            float x1 = L1 * sinf(theta1[i]);
            float y1 = L1 * cosf(theta1[i]);
            float x2 = tip_x[i];
            float y2 = tip_y[i];

            int px1 = cx + (int)(x1 * scale);
            int py1 = cy + (int)(y1 * scale);
            int px2 = cx + (int)(x2 * scale);
            int py2 = cy + (int)(y2 * scale);

            // Simple line drawing for rods
            auto draw_line = [&](int ax, int ay, int bx, int by, Color c) {
                int steps = std::max(abs(bx-ax), abs(by-ay));
                if (steps == 0) return;
                for (int s = 0; s <= steps; s++) {
                    int x = ax + (bx-ax) * s / steps;
                    int y = ay + (by-ay) * s / steps;
                    if (x >= 0 && x < W && y >= 0 && y < H) {
                        int idx = (y * W + x) * 3;
                        fb[idx] = c.r; fb[idx+1] = c.g; fb[idx+2] = c.b;
                    }
                }
            };

            Color rod = {(uint8_t)(colors[i].r/2), (uint8_t)(colors[i].g/2), (uint8_t)(colors[i].b/2)};
            draw_line(cx, cy, px1, py1, rod);
            draw_line(px1, py1, px2, py2, rod);

            // Mass dots
            for (int dy = -3; dy <= 3; dy++)
                for (int dx = -3; dx <= 3; dx++)
                    if (dx*dx+dy*dy <= 9) {
                        int x = px2+dx, y = py2+dy;
                        if (x >= 0 && x < W && y >= 0 && y < H) {
                            int idx = (y*W+x)*3;
                            fb[idx]=colors[i].r; fb[idx+1]=colors[i].g; fb[idx+2]=colors[i].b;
                        }
                    }
        }

        // Pivot dot
        for (int dy = -3; dy <= 3; dy++)
            for (int dx = -3; dx <= 3; dx++)
                if (dx*dx+dy*dy <= 9 && cx+dx >= 0 && cx+dx < W && cy+dy >= 0 && cy+dy < H) {
                    int idx = ((cy+dy)*W+(cx+dx))*3;
                    fb[idx]=200; fb[idx+1]=200; fb[idx+2]=200;
                }

        fwrite(fb.data(), 1, W * H * 3, pipe);

        if (frame % 40 == 0)
            std::cout << "  Frame " << frame << "/" << TOTAL_FRAMES << std::endl;
    }

    pclose(pipe);
    std::cout << "Done! GIF saved to /tmp/pendulum_chaos.gif" << std::endl;
    return 0;
}
