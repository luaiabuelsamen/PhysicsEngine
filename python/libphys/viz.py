"""Small software renderer for libphys worlds and tactile readings.

No OpenGL or display needed: flat-shaded triangles drawn with PIL in
painter's order, which is plenty for the few bodies of an RL env.

    from libphys import viz
    cam = viz.Camera(eye=(0.3, -0.4, 0.3), target=(0, 0, 0))
    img = viz.render(world, env=0, camera=cam)          # PIL.Image
    gel = viz.gelsight(world.tactile[0][0], cell=1e-3)  # simulated sensor image
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from ._libphys import Shape

PALETTE = [(231, 111, 81), (42, 157, 143), (233, 196, 106), (38, 70, 83), (244, 162, 97),
           (118, 120, 237), (214, 40, 57), (106, 153, 78)]


class Camera:
    """Perspective camera looking from `eye` at `target`; fov in degrees."""

    def __init__(self, eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), fov=40.0, size=(480, 360)):
        self.eye = np.asarray(eye, float)
        self.size = size
        f = np.asarray(target, float) - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, float))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.rot = np.stack([r, u, f])  # world -> camera axes (x right, y up, z forward)
        self.focal = 0.5 * size[1] / math.tan(math.radians(fov) / 2)

    def project(self, p):
        """World points [..., 3] -> pixel x, y and depth."""
        c = (np.asarray(p, float) - self.eye) @ self.rot.T
        z = np.maximum(c[..., 2], 1e-6)
        x = self.size[0] / 2 + self.focal * c[..., 0] / z
        y = self.size[1] / 2 - self.focal * c[..., 1] / z
        return x, y, c[..., 2]


# --- meshes in body coordinates ---------------------------------------------------

def _quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _sphere_mesh(r, center=(0, 0, 0), n=14, m=9, y_range=(0.0, math.pi)):
    """Quads of a sphere (or a band of it, polar angle measured from +y)."""
    quads = []
    th = np.linspace(y_range[0], y_range[1], m + 1)
    ph = np.linspace(0, 2 * math.pi, n + 1)
    c = np.asarray(center, float)
    for i in range(m):
        for j in range(n):
            pts = []
            for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                t, p = th[a], ph[b]
                pts.append(c + r * np.array([math.sin(t) * math.cos(p), math.cos(t), math.sin(t) * math.sin(p)]))
            quads.append(np.array(pts))
    return quads


def _box_mesh(h):
    hx, hy, hz = h
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return [v[list(f)] for f in faces]


def _capsule_mesh(r, hl, n=14):
    quads = []
    ph = np.linspace(0, 2 * math.pi, n + 1)
    for j in range(n):
        a, b = ph[j], ph[j + 1]
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        quads.append(np.array([[r * ca, -hl, r * sa], [r * ca, hl, r * sa], [r * cb, hl, r * sb],
                               [r * cb, -hl, r * sb]]))
    quads += _sphere_mesh(r, (0, hl, 0), n, 5, (0, math.pi / 2))
    quads += _sphere_mesh(r, (0, -hl, 0), n, 5, (math.pi / 2, math.pi))
    return quads


def body_quads(shape, size, plane_extent=0.5, plane_tiles=10):
    """Polygons of a body's shape in its own frame, with a per-polygon shade
    factor (planes are drawn as a checkerboard)."""
    if shape == Shape.Sphere:
        q = _sphere_mesh(size[0])
    elif shape == Shape.Capsule:
        q = _capsule_mesh(size[0], size[1])
    elif shape == Shape.Box:
        q = _box_mesh(size)
    elif shape == Shape.Plane:
        q, shades = [], []
        s = np.linspace(-plane_extent, plane_extent, plane_tiles + 1)
        for i in range(plane_tiles):
            for j in range(plane_tiles):
                q.append(np.array([[s[i], 0, s[j]], [s[i], 0, s[j + 1]], [s[i + 1], 0, s[j + 1]],
                                   [s[i + 1], 0, s[j]]]))
                shades.append(1.0 if (i + j) % 2 else 0.88)
        return q, shades
    else:
        return [], []
    return q, [1.0] * len(q)


def render(world, env=0, camera=None, colors=None, background=(246, 244, 239), light=(0.4, -0.5, 0.8),
           plane_extent=0.5, hidden=(), supersample=2, overlays=()):
    """Render one env of a World. colors: {body: (r, g, b)}; hidden: bodies
    to skip; overlays: extra [(points [k, 3], color)] polygons in world
    coordinates (e.g. a tactile map on a pad)."""
    camera = camera or Camera(eye=(1.5, -2.0, 1.2))
    W, H = camera.size
    big = Camera.__new__(Camera)
    big.__dict__.update(camera.__dict__)
    big.size = (W * supersample, H * supersample)
    big.focal = camera.focal * supersample
    s = world.state
    px, py, pz = (t[env].detach().cpu().numpy() for t in (s.px, s.py, s.pz))
    qw, qx, qy, qz = (t[env].detach().cpu().numpy() for t in (s.qw, s.qx, s.qy, s.qz))
    enabled = s.enabled[env].detach().cpu().numpy()
    model = world._world
    bodies = world._bodies
    light = np.asarray(light, float) / np.linalg.norm(light)
    polys = []  # (depth, xy, rgb)
    for b, desc in enumerate(bodies):
        if not enabled[b] or b in hidden:
            continue
        quads, shades = body_quads(desc.shape, (desc.size.x, desc.size.y, desc.size.z), plane_extent)
        if not quads:
            continue
        R = _quat_matrix((qw[b], qx[b], qy[b], qz[b]))
        pos = np.array([px[b], py[b], pz[b]])
        base = np.array((colors or {}).get(b, (200, 200, 200) if desc.shape == Shape.Plane
                                           else PALETTE[b % len(PALETTE)]), float)
        is_plane = desc.shape == Shape.Plane
        for quad, shade in zip(quads, shades):
            w = quad @ R.T + pos
            poly = _shade(big, w, base * shade, light, two_sided=is_plane, inside=None if is_plane else pos)
            if poly is not None and is_plane:
                poly = (poly[0] + 1e6,) + poly[1:]  # floors first: everything rests on them
            polys.append(poly)
    for pts, rgb in overlays:
        polys.append(_shade(big, np.asarray(pts, float), np.asarray(rgb, float), light, two_sided=True,
                            flat=True, bias=-1e-4))
    polys = [p for p in polys if p is not None]
    polys.sort(key=lambda p: -p[0])
    img = Image.new("RGB", big.size, background)
    draw = ImageDraw.Draw(img)
    for _, xy, rgb in polys:
        draw.polygon(xy, fill=rgb)
    return img.resize((W, H), Image.LANCZOS)


def _shade(cam, w, base, light, two_sided=False, flat=False, bias=0.0, inside=None):
    n = np.cross(w[1] - w[0], w[2] - w[0])
    if np.linalg.norm(n) < 1e-14:
        n = np.cross(w[2] - w[0], w[3] - w[0]) if len(w) > 3 else n
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return None
    n /= norm
    centre = w.mean(axis=0)
    if inside is not None and np.dot(n, centre - inside) < 0:
        n = -n  # convex shapes: face normals point away from the body centre
    view = centre - cam.eye
    facing = np.dot(n, view) < 0
    if not facing:
        if not two_sided:
            return None
        n = -n
    x, y, z = cam.project(w)
    if np.any(z <= 1e-4):
        return None
    if flat:
        rgb = base
    else:
        rgb = base * (0.45 + 0.55 * max(0.0, float(np.dot(n, light))))
    rgb = tuple(int(c) for c in np.clip(rgb, 0, 255))
    return float(np.mean(z)) + bias, list(zip(x.tolist(), y.tolist())), rgb


# --- tactile ---------------------------------------------------------------------

def gelsight(reading, cell, scale=8, gain=2.5):
    """A GelSight-style image of a tactile reading ([7, ny, nx] tensor or
    array, lp.TACTILE_CHANNELS): the gel's deflection shaded by three
    coloured lights from the sides, as a vision-based sensor sees its gel.
    cell: cell size (m); gain exaggerates slopes for visibility."""
    from scipy.ndimage import gaussian_filter, zoom
    r = reading.detach().cpu().numpy() if hasattr(reading, "detach") else np.asarray(reading)
    depth = r[3].astype(np.float64)  # deflection into the pad (m)
    # Cubic-spline upsampling plus the blur of the gel's optics.
    big = gaussian_filter(zoom(depth, scale, order=3), sigma=0.6 * scale)
    h = -big
    gy, gx = np.gradient(h, cell / scale)
    nrm = np.stack([-gx * gain, -gy * gain, np.ones_like(h)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    lights = [((1.0, 0.0, 0.6), (235, 60, 60)), ((-0.5, 0.87, 0.6), (60, 210, 90)),
              ((-0.5, -0.87, 0.6), (70, 110, 245))]
    img = np.full(h.shape + (3,), 0.0)
    for d, col in lights:
        d = np.asarray(d) / np.linalg.norm(d)
        lam = np.clip(nrm @ d, 0, 1) - d[2]  # change relative to the flat gel
        img += lam[..., None] * np.asarray(col, float)
    img = np.clip(150 + 1.6 * img, 0, 255).astype(np.uint8)
    return Image.fromarray(img[::-1])


def tactile_map(reading, scale=8, pressure_max=None, arrows=True, stick=True):
    """Pressure (colour), shear (arrows) and the sticking zone (white
    outline) of a tactile reading, as an image."""
    r = reading.detach().cpu().numpy() if hasattr(reading, "detach") else np.asarray(reading)
    p, qx, qy, st = r[0], r[1], r[2], r[6]
    ny, nx = p.shape
    vmax = pressure_max or max(float(p.max()), 1e-9)
    t = np.clip(p / vmax, 0, 1)
    # A dark-to-warm colour ramp.
    stops = np.array([[20, 22, 40], [80, 40, 120], [200, 70, 90], [250, 170, 70], [252, 240, 200]], float)
    idx = t * (len(stops) - 1)
    lo = np.floor(idx).astype(int).clip(0, len(stops) - 2)
    frac = (idx - lo)[..., None]
    rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    img = Image.fromarray(rgb.astype(np.uint8)).resize((nx * scale, ny * scale), Image.NEAREST)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    draw = ImageDraw.Draw(img)
    if stick:
        # Outline the sticking zone: cell edges between stick and not-stick.
        st_in = (p > 0) & (st > 0.5)
        for j in range(ny):
            for i in range(nx):
                if not st_in[j, i]:
                    continue
                x0, y0 = i * scale, (ny - 1 - j) * scale
                if i == 0 or not st_in[j, i - 1]:
                    draw.line([x0, y0, x0, y0 + scale], fill=(255, 255, 255), width=2)
                if i == nx - 1 or not st_in[j, i + 1]:
                    draw.line([x0 + scale, y0, x0 + scale, y0 + scale], fill=(255, 255, 255), width=2)
                if j == ny - 1 or not st_in[j + 1, i]:
                    draw.line([x0, y0, x0 + scale, y0], fill=(255, 255, 255), width=2)
                if j == 0 or not st_in[j - 1, i]:
                    draw.line([x0, y0 + scale, x0 + scale, y0 + scale], fill=(255, 255, 255), width=2)
    if arrows:
        qmax = max(float(np.hypot(qx, qy).max()), 1e-9)
        step = max(1, nx // 12)
        for j in range(step // 2, ny, step):
            for i in range(step // 2, nx, step):
                if p[j, i] <= 0:
                    continue
                L = 2.2 * scale * step * math.hypot(qx[j, i], qy[j, i]) / qmax
                if L < 1:
                    continue
                cx, cy = (i + 0.5) * scale, (ny - j - 0.5) * scale
                a = math.atan2(-qy[j, i], qx[j, i])
                ex, ey = cx + L * math.cos(a), cy + L * math.sin(a)
                draw.line([cx, cy, ex, ey], fill=(120, 230, 255), width=2)
                for da in (2.6, -2.6):
                    draw.line([ex, ey, ex + 4 * math.cos(a + da), ey + 4 * math.sin(a + da)], fill=(120, 230, 255),
                              width=2)
    return img


def label(img, text, xy=(6, 4), color=(30, 30, 30)):
    ImageDraw.Draw(img).text(xy, text, fill=color)
    return img


def save_gif(frames, path, fps=20):
    """Write frames (PIL images of one size) as a looping GIF: through ffmpeg
    with a per-file palette and dithering when available, else with PIL."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        w, h = frames[0].size
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-", "-vf",
               "split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
               "-loop", "0", path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(f.convert("RGB").tobytes())
        proc.stdin.close()
        if proc.wait() == 0:
            return
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
