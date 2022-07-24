"""
Designing a new napari logo
---------------------------

This file contains a proposal for a new napari logo that reflects the original,
itself a reflection of the island of Tabuaeran on which Napari is located.

The new design is based on (what else) the golden ratio ϕ. The two lobes of the
island are represented by two touching circles. The smaller circle's radius is
1/ϕ of the larger circle.

The "neck" of the island is itself shaped by two *outside* circles. The top one
is another factor 1/ϕ smaller, while the bottom one is ϕ bigger. Thus the logo
is defined by a sequence of radii ϕ, ϕ², ϕ³, and ϕ⁴. For mathematical
convenience, in the code the radii used are (1/ϕ)², (1/ϕ), 1, and ϕ. A useful
identity when developing this and to follow the code is 1/ϕ = ϕ - 1. It's also
a very *cool* identity!

Each layer is centered and oriented for mathematical convenience, then
transformed to its final position. The island is centered on the center of the
smaller circle and angled along the 0th axis, which we call x here — it points
down. y points right.

The background is a squircle, |x|⁵ + |y|⁵ = 1, centered on (0, 0) then scaled
and translated.
"""
import numpy as np
import napari
import imageio as iio

# Color constants
# ---------------

LAGOON = np.array((147, 207, 251, 255)) / 255  # "napari blue" from website
# LAGOON = (100, 170, 190)  # from the image itself
# LAGOON = (80, 100, 130)  # from a darker bit from the image
OCEAN = np.array((60, 55, 70, 255)) / 255
SAND = np.array((210, 205, 200, 255)) / 255
FOREST = np.array((60, 90, 25, 255)) / 255

# Numeric constants
# -----------------

# everyone's favorite ratio
phi = (1 + np.sqrt(5)) / 2
# If you make the logo square, put the island at a 45° angle, and margins
# around it of ϕ-1, this is the side length of the square.
source_sidelen = (3 + 1 / np.sqrt(2)) * phi - 2
# But when making a logo we want it to be 1024x1024 or similar. Set the pixel
# size of the square here
target_sidelen = 1024
scale = target_sidelen / source_sidelen
# the margins + the radius of the smaller circle.
translate = 2 * (phi-1) * scale

# A squircle has range (-1, 1) in both x and y, hence an unscaled side length
# of 1 - (-1) = 2.
squircle_scale = target_sidelen / 2
squircle_translate = target_sidelen / 2

# radius and center of the smaller bit of the island
r0 = phi - 1  # = 1/phi
c0 = np.zeros(2)

# radius and center of the larger bit of the island
r1 = 1.
c1 = np.array([phi, 0])

# radius and center of the top of the neck
r2 = 2 - phi  # = (phi - 1)**2
c2 = np.array([xx := 7 - 4*phi, np.sqrt(1 - xx**2)])

# radius and center of the bottom of the neck
r3 = phi
c3 = np.array([2*phi - 3, 2 * np.sqrt(2*phi - 2)])

# Intersection point of the small circle and the outside circles
x1t = c0[0] + r0 / (r0+r2) * (c2[0] - c0[0])
x1b = c0[0] + r0 / (r0+r3) * (c3[0] - c0[0])
# intersection point of the outside circles and the large circle
x2t = c1[0] + r1 / (r1+r2) * (c2[0] - c1[0])
x2b = c1[0] + r1 / (r1+r3) * (c3[0] - c1[0])

# distance from the top of the small circle to the bottom of the large circle
domain = np.array([1 - phi, phi + 1])
xs = np.linspace(*domain, num=1001, endpoint=True)

lines_top = []
lines_bottom = []


def f0(x):
    """Return the half-circle of the smaller lobe of the island."""
    return np.sqrt(np.clip(r0**2 - x**2, 0, None))


def f1(x):
    """Return the half-circle of the larger lobe of the island."""
    return np.sqrt(np.clip(r1**2 - (x - c1[0])**2, 0, None))


def f2(x):
    """Return the half-circle of the smaller neck of the island."""
    return c2[1] - np.sqrt(np.clip(r2**2 - (x - c2[0])**2, 0, None))


def f3(x):
    """Return the half-circle of the smaller neck of the island."""
    return c3[1] - np.sqrt(np.clip(r3**2 - (x - c3[0])**2, 0, None))


# The three segments of the top curve
xs0t = xs[xs < x1t]
xs1t = xs[(x1t <= xs) & (xs < x2t)]
xs2t = xs[x2t <= xs]

# The three segments of the bottom curve (but inverted)
xs0b = xs[xs < x1b]
xs1b = xs[(x1b <= xs) & (xs < x2b)]
xs2b = xs[x2b <= xs]

# The y values (axis=1) for the top curve. Note the numbers are segment
# numbers, not section numbers, hence the confusing inversion between 1 and 2
# with the function names. Sorry.
ys0t = f0(xs0t)
ys1t = f2(xs1t)
ys2t = f1(xs2t)
ys_top = np.concatenate([ys0t, ys1t, ys2t])
line_top = np.stack([xs, ys_top], axis=1)

# As above for the bottom part of the curve.
ys0b = -f0(xs0b)
ys1b = -f3(xs1b)
ys2b = -f1(xs2b)
ys_bottom = np.concatenate([ys0b, ys1b, ys2b])
line_bottom = np.stack([xs[-2:0:-1], ys_bottom[-2:0:-1]], axis=1)

island_shape = np.concatenate([line_top, line_bottom])


def unit_squircle(n_points=4000):
    xs1 = np.linspace(-1, 1, n_points//2 + 1)
    xs2 = xs1[-2:0:-1]
    ys1 = (1 - np.abs(xs1)**5)**0.2
    ys2 = -ys1[-2:0:-1]
    xs = np.concatenate([xs1, xs2])
    ys = np.concatenate([ys1, ys2])
    pts = np.stack([xs, ys], axis=1)
    return pts


def bottom_left_squircle_mask(n_squircle_points=4000):
    pts = unit_squircle(n_points=n_squircle_points)
    top_left_squircle = pts[(pts[:, 1] <= 0) & (pts[:, 0] < 0)]
    bottom_left_squircle = pts[(pts[:, 1] <= 0) * (pts[:, 0] >= 0)]
    top_left = np.array([[-1, -2]])
    bottom_left = np.array([[2, -2]])
    bottom_right = np.array([[2, 0]])
    mask_outline = np.concatenate([
            top_left_squircle, top_left, bottom_left, bottom_right,
            bottom_left_squircle
            ])
    return mask_outline[1:]  # remove discontinuous point at [-1, 0]


def full(r):
    return np.array([r, r])


if __name__ == '__main__':
    viewer = napari.Viewer()
    shp = island_shape
    extra_params = dict(
            scale=[scale, scale],
            translate=[translate, translate],
            rotate=45,
            )
    background_layer = viewer.add_shapes(
            [unit_squircle()],
            shape_type='polygon',
            edge_width=0,
            face_color=OCEAN,
            name='background',
            opacity=1,
            scale=np.full(2, squircle_scale),
            translate=np.full(2, squircle_translate),
            )
    sandbank_layer = viewer.add_shapes(
            [shp],
            shape_type='polygon',
            edge_width=phi / 5,
            edge_color=SAND,
            face_color='transparent',
            name='sand',
            opacity=1,
            **extra_params,
            )
    lagoon_layer = viewer.add_shapes(
            [shp],
            shape_type='polygon',
            edge_width=0,
            face_color=LAGOON,
            name='lagoon',
            opacity=1,
            **extra_params,
            )
    forest_layer = viewer.add_shapes(
            [shp],
            shape_type='polygon',
            edge_width=phi / 10,
            edge_color=FOREST,
            face_color='transparent',
            name='forest',
            opacity=1,
            **extra_params,
            )

    basis_circle_top = viewer.add_shapes(
            [np.stack([c2, full(r2)])],
            shape_type='ellipse',
            edge_width=0.02,
            edge_color=SAND,
            face_color='transparent',
            name='basis-circle-top',
            opacity=0.7,
            visible=False,
            **extra_params,
            )
    basis_circles_inner = viewer.add_shapes(
            [np.stack([c0, full(r0)]),
             np.stack([c1, full(r1)])],
            shape_type='ellipse',
            edge_width=0.02,
            edge_color=SAND,
            face_color='transparent',
            name='basis-circles-inner',
            opacity=0.7,
            visible=False,
            **extra_params,
            )
    basis_circle_bottom = viewer.add_shapes(
            [np.stack([c3 * [1, -1], full(r3)])],
            shape_type='ellipse',
            edge_width=0.02,
            edge_color=SAND,
            face_color='transparent',
            name='basis-circle-bottom',
            opacity=0.7,
            visible=False,
            **extra_params,
            )
    centers = viewer.add_points(
            [c0, c1, c2, c3 * [1, -1]],
            size=0.1,
            face_color='white',
            name='centers',
            visible=False,
            **extra_params,
            )
    contacts = viewer.add_points(
            [
                    [x1t, f0(x1t)],  # 0
                    [x1t, f2(x1t)],  # 1
                    [x2t, f2(x2t)],  # 2
                    [x2t, f1(x2t)],  # 3
                    [x2b, -f1(x2b)],  # 4
                    [x2b, -f3(x2b)],  # 5
                    [x1b, -f3(x1b)],  # 6
                    [x1b, -f0(x1b)],  # 7
                    ],
            size=0.1,
            face_color='red',
            name='contacts',
            visible=False,
            **extra_params,
            )
    mask_layer = viewer.add_shapes(
            [msk := bottom_left_squircle_mask(), msk[:, (1, 0)]],
            shape_type='polygon',
            edge_width=0,
            face_color='black',
            name='mask',
            opacity=1,
            visible=False,
            scale=np.full(2, squircle_scale),
            translate=np.full(2, squircle_translate),
            )

    screenshot = viewer.screenshot()
    mask = np.all([screenshot[:, :, i] == 0 for i in range(3)], axis=0)
    screenshot[mask, 3] = 0
    iio.imsave('logo.png', screenshot)

    napari.run()
