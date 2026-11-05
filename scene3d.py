"""
scene3d.py - builds and renders a seamlessly looping 3D lo-fi scene in Blender:
a cozy desk by a rainy window at night, city lights outside, a sleeping cat.

Everything (geometry, materials, rain, droplets, steam, screen) is generated
procedurally - no downloaded models or textures.

Run with Blender (free, blender.org):
    blender -b -P scene3d.py -- --still 1 --percent 50      # quick preview frame
    blender -b -P scene3d.py --                              # full loop -> render/frames
Options after "--":
    --seconds 12    loop length          --fps 30
    --percent 100   resolution scale      --samples 48   Eevee samples
    --still N       render only frame N to render/still.png
    --seed 3        new city layout / rain pattern
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "render"

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=12)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--percent", type=int, default=100)
ap.add_argument("--samples", type=int, default=48)
ap.add_argument("--still", type=int, default=0)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args(argv)

FPS = args.fps
L = int(round(args.seconds * FPS))     # loop length in frames; frame L+1 == frame 1
rng = np.random.default_rng(args.seed)
TAU = 2 * math.pi


# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"


def keyframes(target, path, func, index=-1, every=1):
    """Key `target.path` on every frame of one loop; func(phase in [0,1]) -> value."""
    for f in list(range(1, L + 1, every)) + [L + 1]:
        val = func((f - 1) / L)
        if index >= 0:
            getattr(target, path)[index] = val
        else:
            setattr(target, path, val)
        target.keyframe_insert(path, index=index, frame=f)


# ---------------------------------------------------------------- textures
def make_image(name, rgba, non_color=False):
    h, w, _ = rgba.shape
    img = bpy.data.images.new(name, w, h, alpha=True)
    if non_color:  # must happen before writing pixels: changing it resets the buffer
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
    img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1], dtype=np.float32).ravel())
    img.update()
    img.pack()
    return img


def periodic_noise(h, w, sigma):
    """Tileable smooth noise in [0,1] (filtered white noise via FFT)."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    k = np.exp(-(fx ** 2 + fy ** 2) * (sigma ** 2) * 20)
    n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) * k))
    return (n - n.min()) / (n.max() - n.min())


def rain_texture(w=256, h=1024, count=140, slant=0.06):
    a = np.zeros((h, w))
    for _ in range(count):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        length = int(rng.uniform(50, 160))
        t = np.arange(length)
        ys = (y0 + t).astype(int) % h
        xs = (x0 + t * slant).astype(int) % w
        np.add.at(a, (ys, xs), rng.uniform(0.25, 0.9) * np.sin(np.pi * t / length))
    a = (a + 0.5 * np.roll(a, 1, axis=1) + 0.5 * np.roll(a, -1, axis=1)) / 2
    a = np.clip(a, 0, 1)
    return np.dstack([np.ones_like(a)] * 3 + [a])


def droplet_texture(n=1024, count=900):
    yy, xx = np.mgrid[0:n, 0:n]
    hmap = np.zeros((n, n))
    for _ in range(count):
        cx, cy = rng.uniform(0, n, 2)
        r = rng.choice([1.5, 2.5, 3.5, 5, 7], p=[0.35, 0.3, 0.2, 0.1, 0.05])
        x0, y0 = int(cx - 3 * r), int(cy - 3 * r)
        size = int(6 * r) + 1
        ys = (np.arange(y0, y0 + size) % n)[:, None]
        xs = (np.arange(x0, x0 + size) % n)[None, :]
        dy = (np.arange(y0, y0 + size)[:, None] - cy) / r
        dx = (np.arange(x0, x0 + size)[None, :] - cx) / (r * 0.85)
        blob = np.clip(1 - (dx ** 2 + dy ** 2), 0, 1) ** 0.6
        hmap[ys, xs] = np.maximum(hmap[ys, xs], blob)
    return hmap


def trail_texture(n=1024, count=26):
    hmap = np.zeros((n, n))
    for _ in range(count):
        x = rng.uniform(0, n)
        y0 = rng.uniform(0, n)
        length = int(rng.uniform(150, 600))
        wob = periodic_noise(1, n, 30)[0] * 14
        for t in range(length):
            y = int(y0 + t) % n
            xc = (x + wob[y]) % n
            fade = (t / length) ** 1.5           # thin tail, fat head
            width = 1.0 + 2.5 * fade
            for dx in range(-4, 5):
                xi = int(xc + dx) % n
                v = max(0, 1 - abs(dx) / width) * (0.4 + 0.6 * fade)
                hmap[y, xi] = max(hmap[y, xi], v)
    return hmap


def window_lights_texture(n=256, cells=16):
    """Grid of building windows, some lit warm/cool. Tileable."""
    img = np.zeros((n, n, 4))
    img[..., 3] = 1
    c = n // cells
    for i in range(cells):
        for j in range(cells):
            if rng.random() < 0.28:
                col = [(1.0, 0.72, 0.4), (1.0, 0.85, 0.6), (0.7, 0.8, 1.0),
                       (1.0, 0.55, 0.35)][rng.integers(4)]
                b = rng.uniform(0.4, 1.0)
                img[i * c + 3:i * c + c - 3, j * c + 4:j * c + c - 4, :3] = np.array(col) * b
    return img


def code_texture(w=512, h=512):
    img = np.zeros((h, w, 4))
    img[..., :3] = (0.035, 0.04, 0.07)
    img[..., 3] = 1
    palette = [(0.55, 0.75, 1.0), (0.95, 0.6, 0.85), (0.6, 0.95, 0.7),
               (1.0, 0.8, 0.45), (0.75, 0.75, 0.8)]
    indent = 0
    for row in range(0, h, 14):
        if rng.random() < 0.12:
            continue
        indent = int(np.clip(indent + rng.choice([-1, 0, 0, 1]), 0, 4))
        x = 24 + indent * 22
        for _ in range(rng.integers(1, 5)):
            seg = int(rng.uniform(20, 90))
            if x + seg > w - 20:
                break
            img[row + 4:row + 10, x:x + seg, :3] = palette[rng.integers(len(palette))]
            x += seg + 8
    return img


# ---------------------------------------------------------------- materials
def material(name, color=(0.8, 0.8, 0.8), rough=0.6, emit=None, strength=1.0, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    if emit:
        p.inputs["Emission Color"].default_value = (*emit, 1)
        p.inputs["Emission Strength"].default_value = strength
    return m


def emissive(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], out.inputs[0])
    return m, e


def transparent(m):
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
        try:
            setattr(m, attr, val)
        except (AttributeError, TypeError):
            pass


def scrolling_uv(nt, image_node, scale=(1, 1), tiles_per_loop=1, axis=1):
    """Mapping node that scrolls the texture by whole tiles over one loop."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (*scale, 1)
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], image_node.inputs["Vector"])
    loc = mp.inputs["Location"]
    loc.default_value[axis] = 0
    loc.keyframe_insert("default_value", index=axis, frame=1)
    loc.default_value[axis] = tiles_per_loop
    loc.keyframe_insert("default_value", index=axis, frame=L + 1)
    return mp


def rain_material(name, img, scale, speed, brightness):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    scrolling_uv(nt, tex, scale, speed)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.62, 0.7, 0.85, 1)
    em.inputs["Strength"].default_value = brightness
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(mix.outputs[0], out.inputs[0])
    transparent(m)
    return m


# ---------------------------------------------------------------- geometry
def box(name, lo, hi, mat, bevel=0.0):
    lo, hi = np.array(lo, float), np.array(hi, float)
    bpy.ops.mesh.primitive_cube_add(size=1, location=tuple((lo + hi) / 2))
    o = bpy.context.object
    o.name = name
    o.scale = tuple(hi - lo)
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    o.data.materials.append(mat)
    return o


def cylinder(name, loc, r, depth, mat, verts=48, rot=(0, 0, 0), **kw):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth,
                                        location=loc, rotation=rot, **kw)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def sphere(name, loc, scale, mat, seg=32):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=seg // 2, radius=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def plane(name, loc, size, mat, rot=(math.pi / 2, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = (size[0], size[1], 1)
    o.data.materials.append(mat)
    return o


def no_shadow(o):
    for attr in ("visible_shadow",):
        try:
            setattr(o, attr, False)
        except AttributeError:
            pass


# ---------------------------------------------------------------- the room
WIN_X = (-1.5, 1.3)
WIN_Z = (0.95, 2.55)
WALL_Y = 2.0

wall_m = material("wall", (0.32, 0.22, 0.2), 0.9)
wood_m = material("wood", (0.25, 0.13, 0.07), 0.45)
frame_m = material("frame", (0.85, 0.82, 0.76), 0.5)
floor_m = material("floor", (0.12, 0.07, 0.05), 0.5)

box("wall_bottom", (-2.5, WALL_Y, 0), (3.5, WALL_Y + 0.15, WIN_Z[0]), wall_m)
box("wall_top", (-2.5, WALL_Y, WIN_Z[1]), (3.5, WALL_Y + 0.15, 3.0), wall_m)
box("wall_left", (-2.5, WALL_Y, WIN_Z[0]), (WIN_X[0], WALL_Y + 0.15, WIN_Z[1]), wall_m)
box("wall_right", (WIN_X[1], WALL_Y, WIN_Z[0]), (3.5, WALL_Y + 0.15, WIN_Z[1]), wall_m)
box("side_wall", (-2.6, -3, 0), (-2.5, WALL_Y + 0.15, 3), wall_m)
box("floor", (-2.6, -3, -0.05), (3.5, WALL_Y, 0), floor_m)
box("ceiling", (-2.6, -3, 3), (3.5, WALL_Y + 0.15, 3.05), wall_m)

t = 0.05  # window frame
box("frame_l", (WIN_X[0], WALL_Y - 0.02, WIN_Z[0]), (WIN_X[0] + t, WALL_Y + 0.12, WIN_Z[1]), frame_m)
box("frame_r", (WIN_X[1] - t, WALL_Y - 0.02, WIN_Z[0]), (WIN_X[1], WALL_Y + 0.12, WIN_Z[1]), frame_m)
box("frame_t", (WIN_X[0], WALL_Y - 0.02, WIN_Z[1] - t), (WIN_X[1], WALL_Y + 0.12, WIN_Z[1]), frame_m)
mid = sum(WIN_X) / 2
box("mullion", (mid - 0.025, WALL_Y + 0.02, WIN_Z[0]), (mid + 0.025, WALL_Y + 0.1, WIN_Z[1]), frame_m)
box("transom", (WIN_X[0], WALL_Y + 0.02, 2.05), (WIN_X[1], WALL_Y + 0.1, 2.09), frame_m)
box("sill", (WIN_X[0] - 0.1, WALL_Y - 0.24, WIN_Z[0] - 0.05), (WIN_X[1] + 0.1, WALL_Y + 0.05, WIN_Z[0]),
    frame_m, bevel=0.01)

# Glass with droplets (static beads + slowly running trails).
glass = bpy.data.materials.new("glass")
glass.use_nodes = True
nt = glass.node_tree
p = nt.nodes["Principled BSDF"]
p.inputs["Base Color"].default_value = (0.05, 0.07, 0.1, 1)
p.inputs["Roughness"].default_value = 0.08
drops = nt.nodes.new("ShaderNodeTexImage")
drops.image = make_image("droplets", np.dstack([droplet_texture()] * 4), non_color=True)
trails = nt.nodes.new("ShaderNodeTexImage")
trails.image = make_image("trails", np.dstack([trail_texture()] * 4), non_color=True)
tc = nt.nodes.new("ShaderNodeTexCoord")
mp = nt.nodes.new("ShaderNodeMapping")
mp.inputs["Scale"].default_value = (2.0, 1.15, 1)
nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
nt.links.new(mp.outputs[0], drops.inputs["Vector"])
scrolling_uv(nt, trails, (2.0, 1.15), 1)
mx = nt.nodes.new("ShaderNodeMath")
mx.operation = "MAXIMUM"
nt.links.new(drops.outputs["Color"], mx.inputs[0])
nt.links.new(trails.outputs["Color"], mx.inputs[1])
alpha = nt.nodes.new("ShaderNodeMapRange")
alpha.inputs["To Min"].default_value = 0.05
alpha.inputs["To Max"].default_value = 0.55
nt.links.new(mx.outputs[0], alpha.inputs["Value"])
nt.links.new(alpha.outputs[0], p.inputs["Alpha"])
bump = nt.nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = 0.8
nt.links.new(mx.outputs[0], bump.inputs["Height"])
nt.links.new(bump.outputs[0], p.inputs["Normal"])
p.inputs["Emission Color"].default_value = (0.5, 0.55, 0.8, 1)
glow = nt.nodes.new("ShaderNodeMath")
glow.operation = "MULTIPLY"
glow.inputs[1].default_value = 0.2
nt.links.new(mx.outputs[0], glow.inputs[0])
nt.links.new(glow.outputs[0], p.inputs["Emission Strength"])
transparent(glass)
g = plane("glass", (mid, WALL_Y + 0.06, sum(WIN_Z) / 2), (WIN_X[1] - WIN_X[0], WIN_Z[1] - WIN_Z[0]), glass)
no_shadow(g)

# ---------------------------------------------------------------- desk + props
box("desk", (-1.35, 1.0, 0.72), (1.25, 1.78, 0.76), wood_m, bevel=0.008)
for x in (-1.3, 1.2):
    for y in (1.05, 1.72):
        box("leg", (x - 0.025, y - 0.025, 0), (x + 0.025, y + 0.025, 0.72), wood_m)

# laptop
alu = material("aluminium", (0.55, 0.56, 0.6), 0.35, metal=0.8)
box("laptop_base", (0.08, 1.22, 0.76), (0.42, 1.46, 0.775), alu, bevel=0.004)
bpy.ops.object.empty_add(location=(0.25, 1.46, 0.775))
hinge = bpy.context.object
screen_back = box("laptop_lid", (0.08, 1.46, 0.775), (0.42, 1.47, 0.995), alu, bevel=0.004)
screen_m, screen_em = emissive("screen", (1, 1, 1), 1.6)
code_tex = screen_m.node_tree.nodes.new("ShaderNodeTexImage")
code_tex.image = make_image("code", code_texture())
screen_m.node_tree.links.new(code_tex.outputs["Color"], screen_em.inputs["Color"])
scrolling_uv(screen_m.node_tree, code_tex, (1, 0.65), 1)
scr = plane("laptop_screen", (0.25, 1.4585, 0.885), (0.32, 0.2), screen_m)
bpy.context.view_layer.update()
for o in (screen_back, scr):
    o.parent = hinge
    o.matrix_parent_inverse = hinge.matrix_world.inverted()  # keep world placement
hinge.rotation_euler = (math.radians(-14), 0, 0)

# desk lamp
lamp_m = material("lamp", (0.15, 0.28, 0.22), 0.4)
cylinder("lamp_base", (-0.85, 1.55, 0.775), 0.09, 0.03, lamp_m)
cylinder("lamp_pole", (-0.85, 1.55, 1.0), 0.012, 0.45, lamp_m)
cylinder("lamp_arm", (-0.78, 1.5, 1.22), 0.01, 0.17, lamp_m, rot=(math.radians(35), math.radians(55), 0))
bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=0.13, radius2=0.04, depth=0.15,
                                location=(-0.7, 1.45, 1.2), end_fill_type="NOTHING")
shade = bpy.context.object
shade.data.materials.append(lamp_m)
shade.modifiers.new("solid", "SOLIDIFY").thickness = 0.006
bpy.ops.object.shade_smooth()
bulb_m, _ = emissive("bulb", (1.0, 0.75, 0.45), 25)
sphere("bulb", (-0.7, 1.45, 1.16), (0.03, 0.03, 0.03), bulb_m)

# mug with steam
mug_m = material("mug", (0.85, 0.83, 0.78), 0.3)
cylinder("mug", (0.78, 1.22, 0.815), 0.045, 0.11, mug_m)
bpy.ops.mesh.primitive_torus_add(major_radius=0.03, minor_radius=0.008,
                                 location=(0.83, 1.22, 0.82), rotation=(math.pi / 2, 0, 0))
bpy.context.object.data.materials.append(mug_m)
coffee_m = material("coffee", (0.08, 0.04, 0.02), 0.1)
cylinder("coffee", (0.78, 1.22, 0.86), 0.041, 0.004, coffee_m)

steam = bpy.data.materials.new("steam")
steam.use_nodes = True
nt = steam.node_tree
nt.nodes.clear()
sn = periodic_noise(256, 256, 18)
sn = np.clip((sn - 0.45) * 2.2, 0, 1)
stex = nt.nodes.new("ShaderNodeTexImage")
stex.image = make_image("steam_noise", np.dstack([sn] * 4), non_color=True)
scrolling_uv(nt, stex, (1, 1.4), -2)
grad = nt.nodes.new("ShaderNodeTexCoord")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(grad.outputs["UV"], sep.inputs[0])
fade = nt.nodes.new("ShaderNodeMath")       # fade toward the top and sides
fade.operation = "MULTIPLY"
up = nt.nodes.new("ShaderNodeMapRange")
up.inputs["From Min"].default_value = 1.0
up.inputs["From Max"].default_value = 0.1
nt.links.new(sep.outputs["Y"], up.inputs["Value"])
side = nt.nodes.new("ShaderNodeMath")
side.operation = "PINGPONG"
side.inputs[1].default_value = 0.5
nt.links.new(sep.outputs["X"], side.inputs[0])
side2 = nt.nodes.new("ShaderNodeMath")
side2.operation = "MULTIPLY"
side2.inputs[1].default_value = 2.0
nt.links.new(side.outputs[0], side2.inputs[0])
nt.links.new(up.outputs[0], fade.inputs[0])
nt.links.new(side2.outputs[0], fade.inputs[1])
dens = nt.nodes.new("ShaderNodeMath")
dens.operation = "MULTIPLY"
nt.links.new(stex.outputs["Color"], dens.inputs[0])
nt.links.new(fade.outputs[0], dens.inputs[1])
dens2 = nt.nodes.new("ShaderNodeMath")
dens2.operation = "MULTIPLY"
dens2.inputs[1].default_value = 0.35
nt.links.new(dens.outputs[0], dens2.inputs[0])
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1.0, 0.85, 0.7, 1)
em.inputs["Strength"].default_value = 0.6
tr = nt.nodes.new("ShaderNodeBsdfTransparent")
mix = nt.nodes.new("ShaderNodeMixShader")
nt.links.new(dens2.outputs[0], mix.inputs["Fac"])
nt.links.new(tr.outputs[0], mix.inputs[1])
nt.links.new(em.outputs[0], mix.inputs[2])
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(mix.outputs[0], out.inputs[0])
transparent(steam)
st = plane("steam", (0.78, 1.22, 1.03), (0.16, 0.34), steam, rot=(math.pi / 2, 0, math.radians(-12)))
no_shadow(st)

# books
for k, (col, h) in enumerate([((0.45, 0.12, 0.1), 0.035), ((0.12, 0.25, 0.4), 0.03),
                               ((0.75, 0.6, 0.3), 0.04)]):
    z0 = 0.76 + sum(x[1] for x in [((0, 0, 0), 0.035), ((0, 0, 0), 0.03)][:k])
    b = box(f"book{k}", (-0.42, 1.45, z0), (-0.12, 1.68, z0 + h), material(f"book{k}", col, 0.7), bevel=0.003)
    b.rotation_euler[2] = rng.uniform(-0.12, 0.12)

# plant
pot_m = material("pot", (0.6, 0.3, 0.2), 0.8)
PX, PY = -1.15, 1.35
cylinder("pot", (PX, PY, 0.82), 0.07, 0.12, pot_m)
leaf_m = material("leaf", (0.12, 0.35, 0.15), 0.5)
for k in range(9):
    ang = k / 9 * TAU + rng.uniform(-0.2, 0.2)
    tilt = rng.uniform(0.5, 1.0)
    lf = sphere(f"leaf{k}", (PX + 0.07 * math.cos(ang), PY + 0.07 * math.sin(ang), 0.95 + rng.uniform(0, 0.06)),
                (0.03, 0.09, 0.008), leaf_m, seg=16)
    lf.rotation_euler = (tilt * math.cos(ang + math.pi / 2), tilt * math.sin(ang + math.pi / 2), ang + math.pi / 2)

# sleeping cat on the windowsill (breathes once every 4 seconds)
cat_m = material("cat", (0.75, 0.42, 0.2), 0.8)
CAT_X = 0.85
cat_body = sphere("cat_body", (CAT_X, 1.88, 1.04), (0.2, 0.12, 0.09), cat_m)
sphere("cat_head", (CAT_X - 0.22, 1.84, 1.07), (0.075, 0.07, 0.065), cat_m)
for dx in (-0.035, 0.035):
    bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=0.025, depth=0.05,
                                    location=(CAT_X - 0.22 + dx, 1.85, 1.135), rotation=(0, dx * 6, 0))
    bpy.context.object.data.materials.append(cat_m)
bpy.ops.mesh.primitive_torus_add(major_radius=0.13, minor_radius=0.022, location=(CAT_X - 0.03, 1.8, 0.99),
                                 rotation=(0, 0, 0))
tail = bpy.context.object
tail.scale = (1.2, 0.7, 1)
tail.data.materials.append(cat_m)
breaths = max(1, round(args.seconds / 4))
keyframes(cat_body, "scale", lambda ph: 0.09 * (1 + 0.04 * math.sin(TAU * breaths * ph)), index=2)

# string lights along the top of the window
fairy = []
for k in range(3):
    m, e = emissive(f"fairy{k}", (1.0, 0.7, 0.35), 8)
    off, rate = k / 3, (1, 2, 1)[k]
    keyframes(e.inputs["Strength"], "default_value",
              lambda ph, off=off, rate=rate: 5 + 4 * math.sin(TAU * (rate * ph + off)), every=2)
    fairy.append(m)
n_bulbs = 22
for k in range(n_bulbs):
    u = k / (n_bulbs - 1)
    x = WIN_X[0] - 0.05 + u * (WIN_X[1] - WIN_X[0] + 0.1)
    z = WIN_Z[1] + 0.06 - 0.12 * math.sin(math.pi * ((u * 2) % 1))
    sphere(f"fairy{k}", (x, WALL_Y - 0.05, z), (0.012, 0.012, 0.016), fairy[k % 3], seg=12)

# ---------------------------------------------------------------- the city outside
fog_col = (0.09, 0.08, 0.15)
bld = bpy.data.materials.new("building")
bld.use_nodes = True
nt = bld.node_tree
p = nt.nodes["Principled BSDF"]
p.inputs["Roughness"].default_value = 0.8
tc = nt.nodes.new("ShaderNodeTexCoord")
oi = nt.nodes.new("ShaderNodeObjectInfo")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Object"], sep.inputs[0])
uadd = nt.nodes.new("ShaderNodeMath")
uadd.operation = "ADD"
nt.links.new(sep.outputs["X"], uadd.inputs[0])
nt.links.new(sep.outputs["Y"], uadd.inputs[1])
rnd = nt.nodes.new("ShaderNodeMath")
rnd.operation = "MULTIPLY"
rnd.inputs[1].default_value = 997
nt.links.new(oi.outputs["Random"], rnd.inputs[0])
uadd2 = nt.nodes.new("ShaderNodeMath")
uadd2.operation = "ADD"
nt.links.new(uadd.outputs[0], uadd2.inputs[0])
nt.links.new(rnd.outputs[0], uadd2.inputs[1])
comb = nt.nodes.new("ShaderNodeCombineXYZ")
nt.links.new(uadd2.outputs[0], comb.inputs["X"])
nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
scale = nt.nodes.new("ShaderNodeVectorMath")
scale.operation = "SCALE"
scale.inputs["Scale"].default_value = 1 / 26  # ~1.6 m window cells, 16 per tile
nt.links.new(comb.outputs[0], scale.inputs[0])
wtex = nt.nodes.new("ShaderNodeTexImage")
wtex.image = make_image("windows", window_lights_texture())
wtex.interpolation = "Closest"
nt.links.new(scale.outputs[0], wtex.inputs["Vector"])
cam_d = nt.nodes.new("ShaderNodeCameraData")
fogf = nt.nodes.new("ShaderNodeMath")
fogf.operation = "DIVIDE"
fogf.inputs[1].default_value = 130
nt.links.new(cam_d.outputs["View Distance"], fogf.inputs[0])
fogc = nt.nodes.new("ShaderNodeMath")
fogc.operation = "MINIMUM"
fogc.inputs[1].default_value = 0.85
nt.links.new(fogf.outputs[0], fogc.inputs[0])
body = nt.nodes.new("ShaderNodeMix")
body.data_type = "RGBA"
body.inputs["A"].default_value = (0.02, 0.022, 0.035, 1)
body.inputs["B"].default_value = (*fog_col, 1)
nt.links.new(fogc.outputs[0], body.inputs["Factor"])
nt.links.new(body.outputs["Result"], p.inputs["Base Color"])
p.inputs["Emission Color"].default_value = (1, 1, 1, 1)
nt.links.new(wtex.outputs["Color"], p.inputs["Emission Color"])
inv = nt.nodes.new("ShaderNodeMapRange")
inv.inputs["From Min"].default_value = 0
inv.inputs["From Max"].default_value = 1
inv.inputs["To Min"].default_value = 3.5
inv.inputs["To Max"].default_value = 0.6
nt.links.new(fogc.outputs[0], inv.inputs["Value"])
nt.links.new(inv.outputs[0], p.inputs["Emission Strength"])

red_lights = []
for k in range(110):
    y = rng.uniform(40, 200)
    x = rng.uniform(-0.75, 0.75) * (y + 2) + 0.2
    w, d = rng.uniform(7, 20), rng.uniform(7, 20)
    top = rng.uniform(-18, 0.25 * y + 4)
    o = box(f"bldg{k}", (x - w / 2, y - d / 2, -90), (x + w / 2, y + d / 2, top), bld)
    no_shadow(o)
    if top > 0.2 * y and len(red_lights) < 5:
        red_lights.append((x, y, top + 1.5))

for k, (x, y, z) in enumerate(red_lights):
    m, e = emissive(f"beacon{k}", (1.0, 0.08, 0.05), 0)
    rate, off = (3, 2, 4, 3, 2)[k], rng.random()
    keyframes(e.inputs["Strength"], "default_value",
              lambda ph, rate=rate, off=off: 40.0 if ((rate * ph + off) % 1) < 0.1 else 0.5)
    sphere(f"beacon{k}", (x, y, z), (0.5, 0.5, 0.5), m, seg=12)

# sky: dark navy fading to a hazy city glow on the horizon
sky_m, sky_em = emissive("sky", (1, 1, 1), 1.0)
nt = sky_m.node_tree
tc = nt.nodes.new("ShaderNodeTexCoord")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Generated"], sep.inputs[0])
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.35
ramp.color_ramp.elements[0].color = (0.16, 0.09, 0.16, 1)
ramp.color_ramp.elements[1].position = 0.75
ramp.color_ramp.elements[1].color = (0.012, 0.016, 0.04, 1)
nt.links.new(sep.outputs["Y"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], sky_em.inputs["Color"])
plane("sky", (0, 220, 20), (700, 300), sky_m)

# rain: three scrolling sheets at different depths for parallax
for k, (y, w, h, speed, bright, scl) in enumerate([
        (3.2, 8, 5, 40, 0.28, (3, 2)), (9, 20, 12, 30, 0.2, (5, 3)), (25, 60, 35, 18, 0.13, (9, 5))]):
    img = make_image(f"rain{k}", rain_texture(count=110 + 40 * k), non_color=True)
    r = plane(f"rain{k}", (0.2, y, 1.6), (w, h), rain_material(f"rain{k}", img, scl, speed, bright))
    no_shadow(r)

# ---------------------------------------------------------------- lights + world
def light(name, kind, loc, energy, color, rot=(0, 0, 0), size=0.05, size_y=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if kind == "AREA":
        ld.shape = "RECTANGLE"
        ld.size = size
        ld.size_y = size_y or size
    else:
        ld.shadow_soft_size = size
    o = bpy.data.objects.new(name, ld)
    o.location = loc
    o.rotation_euler = rot
    scene.collection.objects.link(o)
    return o


light("lamp", "POINT", (-0.7, 1.45, 1.14), 70, (1.0, 0.62, 0.32), size=0.04)
light("wall_wash", "POINT", (-0.7, 1.7, 1.35), 25, (1.0, 0.6, 0.35), size=0.2)
light("window_fill", "AREA", (mid, 1.9, 1.75), 25, (0.45, 0.55, 1.0),
      rot=(math.radians(-90), 0, 0), size=2.8, size_y=1.6)
light("screen_glow", "AREA", (0.25, 1.40, 0.9), 3, (0.6, 0.72, 1.0),
      rot=(math.radians(-104), 0, 0), size=0.3, size_y=0.2)
light("room_fill", "AREA", (1.5, -1.5, 2.6), 15, (1.0, 0.7, 0.5),
      rot=(math.radians(-50), 0, math.radians(30)), size=2)

world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.006, 0.007, 0.014, 1)

# ---------------------------------------------------------------- camera
bpy.ops.object.empty_add(location=(-0.08, 1.8, 1.24))
target = bpy.context.object
cam_d = bpy.data.cameras.new("cam")
cam_d.lens = 30
cam_d.dof.use_dof = True
cam_d.dof.aperture_fstop = 1.8
cam = bpy.data.objects.new("cam", cam_d)
scene.collection.objects.link(cam)
cam.location = (0.3, -0.8, 1.45)
focus = np.array((0.25, 1.4, 0.88))  # the laptop screen
cam_d.dof.focus_distance = float(np.linalg.norm(focus - np.array(cam.location)))
c = cam.constraints.new("TRACK_TO")
c.target = target
c.track_axis = "TRACK_NEGATIVE_Z"
c.up_axis = "UP_Y"
scene.camera = cam
base = cam.location.copy()
keyframes(cam, "location", lambda ph: base[0] + 0.03 * math.sin(TAU * ph), index=0, every=3)
keyframes(cam, "location", lambda ph: base[2] + 0.012 * math.sin(TAU * 2 * ph + 1), index=2, every=3)

# ---------------------------------------------------------------- render settings
r = scene.render
r.engine = "BLENDER_EEVEE"
r.resolution_x, r.resolution_y = 1920, 1080
r.resolution_percentage = args.percent
r.fps = FPS
scene.frame_start, scene.frame_end = 1, L
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGB"
ee = scene.eevee
for attr, val in (("taa_render_samples", args.samples), ("use_shadows", True),
                  ("use_raytracing", True), ("use_gtao", True)):
    try:
        setattr(ee, attr, val)
    except AttributeError:
        pass
try:
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    pass
scene.view_settings.exposure = 0.8

def all_fcurves(action):
    try:
        yield from action.fcurves                 # Blender < 5
    except AttributeError:
        pass
    for layer in getattr(action, "layers", []):   # Blender 4.4+ layered actions
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


# Constant-speed motion is what makes the loop seamless; the default
# Bezier easing would make the rain slow down and stop at the loop point.
n_keys = 0
for action in bpy.data.actions:
    for fc in all_fcurves(action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
            n_keys += 1
print(f"linearized {n_keys} keyframes")

OUT.mkdir(exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "lofi_room.blend"))

if args.still:
    scene.frame_set(args.still)
    r.filepath = str(OUT / "still.png")
    bpy.ops.render.render(write_still=True)
else:
    (OUT / "frames").mkdir(exist_ok=True)
    r.filepath = str(OUT / "frames" / "f_")
    bpy.ops.render.render(animation=True)
print("DONE")
