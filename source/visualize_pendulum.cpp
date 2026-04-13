// visualize_pendulum.cpp
// Renders multiple double pendulums with slightly different initial conditions
// to show chaotic divergence. Outputs GIF via ffmpeg pipe.
// Pure CPU - double pendulum is only 4 ODEs, no GPU needed.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <deque>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Color { uint8_t r, g, b; };

struct PendulumState {
    double theta1, theta2;   // angles
    double omega1, omega2;   // angular velocities
};

struct TrailPoint { int x, y; };

// Double pendulum equations of motion
static void derivatives(const PendulumState& s, double g, double L1, double L2,
                          double m1, double m2,
                          double& dtheta1, double& dtheta2,
                          double& domega1, double& domega2) {
    double dt = s.theta1 - s.theta2;
    double sin_dt = sin(dt);
    double cos_dt = cos(dt);
    double denom = 2*m1 + m2 - m2*cos(2*dt);

    dtheta1 = s.omega1;
    dtheta2 = s.omega2;

    domega1 = (-g*(2*m1+m2)*sin(s.theta1) - m2*g*sin(s.theta1-2*s.theta2)
               - 2*sin_dt*m2*(s.omega2*s.omega2*L2 + s.omega1*s.omega1*L1*cos_dt))
              / (L1 * denom);

    domega2 = (2*sin_dt*(s.omega1*s.omega1*L1*(m1+m2) + g*(m1+m2)*cos(s.theta1)
               + s.omega2*s.omega2*L2*m2*cos_dt))
              / (L2 * denom);
}

// RK4 step
static PendulumState rk4_step(const PendulumState& s, double dt_step,
                                double g, double L1, double L2, double m1, double m2) {
    double k1t1, k1t2, k1o1, k1o2;
    double k2t1, k2t2, k2o1, k2o2;
    double k3t1, k3t2, k3o1, k3o2;
    double k4t1, k4t2, k4o1, k4o2;

    derivatives(s, g, L1, L2, m1, m2, k1t1, k1t2, k1o1, k1o2);

    PendulumState s2 = {s.theta1 + 0.5*dt_step*k1t1, s.theta2 + 0.5*dt_step*k1t2,
                        s.omega1 + 0.5*dt_step*k1o1, s.omega2 + 0.5*dt_step*k1o2};
    derivatives(s2, g, L1, L2, m1, m2, k2t1, k2t2, k2o1, k2o2);

    PendulumState s3 = {s.theta1 + 0.5*dt_step*k2t1, s.theta2 + 0.5*dt_step*k2t2,
                        s.omega1 + 0.5*dt_step*k2o1, s.omega2 + 0.5*dt_step*k2o2};
    derivatives(s3, g, L1, L2, m1, m2, k3t1, k3t2, k3o1, k3o2);

    PendulumState s4 = {s.theta1 + dt_step*k3t1, s.theta2 + dt_step*k3t2,
                        s.omega1 + dt_step*k3o1, s.omega2 + dt_step*k3o2};
    derivatives(s4, g, L1, L2, m1, m2, k4t1, k4t2, k4o1, k4o2);

    return {
        s.theta1 + dt_step/6.0*(k1t1 + 2*k2t1 + 2*k3t1 + k4t1),
        s.theta2 + dt_step/6.0*(k1t2 + 2*k2t2 + 2*k3t2 + k4t2),
        s.omega1 + dt_step/6.0*(k1o1 + 2*k2o1 + 2*k3o1 + k4o1),
        s.omega2 + dt_step/6.0*(k1o2 + 2*k2o2 + 2*k3o2 + k4o2)
    };
}

static void draw_line(std::vector<uint8_t>& fb, int W, int H,
                       int x0, int y0, int x1, int y1, Color c, int thickness = 2) {
    int dx = abs(x1 - x0), dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        for (int ty = -thickness/2; ty <= thickness/2; ty++) {
            for (int tx = -thickness/2; tx <= thickness/2; tx++) {
                int px = x0 + tx, py = y0 + ty;
                if (px >= 0 && px < W && py >= 0 && py < H) {
                    int idx = (py * W + px) * 3;
                    fb[idx] = c.r; fb[idx+1] = c.g; fb[idx+2] = c.b;
                }
            }
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

static void draw_filled_circle(std::vector<uint8_t>& fb, int W, int H,
                                int cx, int cy, int radius, Color c) {
    for (int dy = -radius; dy <= radius; dy++) {
        int y = cy + dy;
        if (y < 0 || y >= H) continue;
        int dx_max = (int)sqrtf((float)(radius*radius - dy*dy));
        for (int dx = -dx_max; dx <= dx_max; dx++) {
            int x = cx + dx;
            if (x < 0 || x >= W) continue;
            int idx = (y * W + x) * 3;
            float shade = 1.0f - 0.25f * sqrtf((float)(dx*dx+dy*dy)) / radius;
            fb[idx]   = (uint8_t)(c.r * shade);
            fb[idx+1] = (uint8_t)(c.g * shade);
            fb[idx+2] = (uint8_t)(c.b * shade);
        }
    }
}

// Draw trail with fading alpha
static void draw_trail(std::vector<uint8_t>& fb, int W, int H,
                        const std::deque<TrailPoint>& trail, Color c) {
    int n = trail.size();
    for (int i = 1; i < n; i++) {
        float alpha = (float)i / n;  // fade in
        Color faded = {(uint8_t)(c.r * alpha * 0.6f),
                       (uint8_t)(c.g * alpha * 0.6f),
                       (uint8_t)(c.b * alpha * 0.6f)};
        // Just draw the point, not a line (faster and looks like a trail)
        int x = trail[i].x, y = trail[i].y;
        // Draw a 2px dot for visible trail
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int px = x + dx, py = y + dy;
                if (px >= 0 && px < W && py >= 0 && py < H) {
                    int idx = (py * W + px) * 3;
                    fb[idx]   = std::min(255, fb[idx]   + (int)faded.r);
                    fb[idx+1] = std::min(255, fb[idx+1] + (int)faded.g);
                    fb[idx+2] = std::min(255, fb[idx+2] + (int)faded.b);
                }
            }
        }
    }
}

int main() {
    const int W = 480, H = 360;
    const int NUM_PENDULUMS = 5;
    const int TOTAL_FRAMES = 250;
    const int SUBSTEPS = 20;
    const double DT = 0.005;
    const int MAX_TRAIL = 1200;

    // Physics params
    double g = 9.81, L1 = 1.0, L2 = 1.0, m1 = 1.0, m2 = 1.0;

    // Initial conditions - very slightly different to show chaos
    std::vector<PendulumState> states(NUM_PENDULUMS);
    double base_theta1 = M_PI * 0.6;
    double base_theta2 = M_PI * 0.4;
    for (int i = 0; i < NUM_PENDULUMS; i++) {
        states[i] = {base_theta1 + (i - NUM_PENDULUMS/2) * 0.01, base_theta2, 0, 0};
    }

    // Colors for each pendulum
    Color colors[] = {
        {255, 50, 50},    // red
        {50, 255, 100},   // green
        {50, 150, 255},   // blue
        {255, 220, 30},   // yellow
        {255, 100, 220},  // pink
    };

    // Trails for tip of second pendulum
    std::vector<std::deque<TrailPoint>> trails(NUM_PENDULUMS);

    // Coordinate mapping: pivot in upper-center, scale so pendulum fills frame nicely
    int cx = W / 2, cy = H * 2 / 7;
    double max_extent = L1 + L2;
    double available = std::min((double)(H - cy - 15), (double)(cx - 15));
    double scale = available * 0.85 / max_extent;

    std::vector<uint8_t> fb(W * H * 3);

    std::string cmd = "ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 480x360 -r 30 "
                      "-i pipe:0 -vf \"fps=24,split[s0][s1];"
                      "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
                      "[s1][p]paletteuse=dither=floyd_steinberg\" "
                      "-loop 0 /tmp/double_pendulum.gif 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) { std::cerr << "Failed to open ffmpeg\n"; return 1; }

    std::cout << "Rendering " << TOTAL_FRAMES << " frames, "
              << NUM_PENDULUMS << " pendulums..." << std::endl;

    for (int frame = 0; frame < TOTAL_FRAMES; frame++) {
        // Substep physics
        for (int s = 0; s < SUBSTEPS; s++) {
            for (int p = 0; p < NUM_PENDULUMS; p++) {
                states[p] = rk4_step(states[p], DT, g, L1, L2, m1, m2);
            }
        }

        // Clear to dark background
        for (int i = 0; i < W * H; i++) {
            fb[i*3] = 10; fb[i*3+1] = 12; fb[i*3+2] = 18;
        }

        // Record trail points and draw trails
        for (int p = 0; p < NUM_PENDULUMS; p++) {
            double x1 = L1 * sin(states[p].theta1);
            double y1 = L1 * cos(states[p].theta1);
            double x2 = x1 + L2 * sin(states[p].theta2);
            double y2 = y1 + L2 * cos(states[p].theta2);

            int tip_x = cx + (int)(x2 * scale);
            int tip_y = cy + (int)(y2 * scale);

            trails[p].push_back({tip_x, tip_y});
            if ((int)trails[p].size() > MAX_TRAIL) trails[p].pop_front();

            draw_trail(fb, W, H, trails[p], colors[p]);
        }

        // Draw pendulums (on top of trails)
        for (int p = 0; p < NUM_PENDULUMS; p++) {
            double x1 = L1 * sin(states[p].theta1);
            double y1 = L1 * cos(states[p].theta1);
            double x2 = x1 + L2 * sin(states[p].theta2);
            double y2 = y1 + L2 * cos(states[p].theta2);

            int px1 = cx + (int)(x1 * scale);
            int py1 = cy + (int)(y1 * scale);
            int px2 = cx + (int)(x2 * scale);
            int py2 = cy + (int)(y2 * scale);

            // Draw rods
            Color rod_col = {(uint8_t)(colors[p].r/2), (uint8_t)(colors[p].g/2), (uint8_t)(colors[p].b/2)};
            draw_line(fb, W, H, cx, cy, px1, py1, rod_col, 2);
            draw_line(fb, W, H, px1, py1, px2, py2, rod_col, 2);

            // Draw masses
            draw_filled_circle(fb, W, H, px1, py1, 7, colors[p]);
            draw_filled_circle(fb, W, H, px2, py2, 9, colors[p]);
        }

        // Draw pivot
        draw_filled_circle(fb, W, H, cx, cy, 4, {200, 200, 200});

        fwrite(fb.data(), 1, W * H * 3, pipe);

        if (frame % 50 == 0)
            std::cout << "  Frame " << frame << "/" << TOTAL_FRAMES << std::endl;
    }

    pclose(pipe);
    std::cout << "Done! GIF saved to /tmp/double_pendulum.gif" << std::endl;
    return 0;
}
