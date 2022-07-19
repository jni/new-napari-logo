import itertools
import numpy as np
import napari

LAGOON = np.array((147, 207, 251, 255)) / 255  # "napari blue" from website
# LAGOON = (100, 170, 190)  # from the image itself
# LAGOON = (80, 100, 130)  # from a darker bit from the image
OCEAN = np.array((40, 50, 70, 255)) / 255
SAND = np.array((210, 205, 200, 255)) / 255
FOREST = np.array((60, 90, 25, 255)) / 255

phi = (1 + np.sqrt(5)) / 2

r0 = phi - 1
c0 = np.zeros(2)
r1 = 1.
c1 = np.array([phi, 0])
r3s = np.array([phi - 1, 1, phi])
c3s = np.array([
        [4*phi - 6, 2 * np.sqrt(10*phi - 3 * phi**2 - 8)],
        [2 - phi, 2 * np.sqrt(phi - 1)],
        [2*phi - 3, 2 * np.sqrt(2*phi - 2)],
        ])

x1s = r0 * c3s[:, 0] / c3s[:, 1]
x2s = c1[0] + r1 / (r1+r3s) * (c3s[:, 0] - c1[0])

domain = np.array([1 - phi, phi + 1])
xs = np.linspace(*domain, num=500, endpoint=True)

lines_top = []
lines_bottom = []


def f0(x):
    return np.sqrt(np.clip(r0**2 - x**2, 0, None))


def f1(x):
    return np.sqrt(np.clip(r1**2 - (x - c1[0])**2, 0, None))


def f3(x, c, r):
    return c[1] - np.sqrt(np.clip(r**2 - (x - c[0])**2, 0, None))


for r3, c3, x1, x2 in zip(r3s, c3s, x1s, x2s):
    xs0 = xs[xs < x1]
    xs1 = xs[(x1 <= xs) & (xs < x2)]
    xs2 = xs[x2 <= xs]

    ys0 = f0(xs0)
    ys1 = f3(xs1, c3, r3)
    ys2 = f1(xs2)

    ys_top = np.concatenate([ys0, ys1, ys2])
    line_top = np.stack([xs, ys_top], axis=1)
    lines_top.append(line_top)

    line_bottom = np.copy(line_top[-2:0:-1])
    line_bottom[:, 1] *= -1

    lines_bottom.append(line_bottom)

candidate_shapes = [  # yapf: ignore
        np.concatenate((line_top, line_bottom), axis=0)
        for line_top, line_bottom in itertools.product(lines_top, lines_bottom)
        ]


def full(r):
    return np.array([r, r])


viewer = napari.Viewer()
for k, shp in enumerate(candidate_shapes[2:3], start=2):
    i, j = np.divmod(k, 3)
    c3i = c3s[i]
    r3i = r3s[i]
    x1i = x1s[i]
    x2i = x2s[i]
    c3j = c3s[j] * [1, -1]
    r3j = r3s[j]
    x1j = x1s[j]
    x2j = x2s[j]
    extra_params = dict(
            scale=[500, 500],
            translate=[100, 100],
            rotate=45,
            )
    viewer.add_shapes(
            [shp, shp],
            shape_type='polygon',
            edge_width=[0.2, 0.1],
            edge_color=[SAND, FOREST],
            face_color=[LAGOON, LAGOON],
            name=f'option-{i}-{j}',
            opacity=1,
            **extra_params,
            )
    viewer.add_shapes(
            [
                    np.stack([c0, full(r0)]),
                    np.stack([c1, full(r1)]),
                    np.stack([c3i, full(r3i)]),
                    np.stack([c3j, full(r3j)]),
                    ],
            shape_type='ellipse',
            edge_width=0.1,
            edge_color='white',
            face_color='transparent',
            opacity=0.5,
            name=f'basis-circles-{i}-{j}',
            **extra_params,
            )
    viewer.add_points(
            [c0, c1, c3i, c3j],
            size=0.1,
            face_color='white',
            name=f'centers-{i}-{j}',
            **extra_params,
            )
    viewer.add_points(
            [
                    [x1i, f0(x1i)],  # 0
                    [x1i, f3(x1i, c3i, r3i)],  # 1
                    [x2i, f3(x2i, c3i, r3i)],  # 2
                    [x2i, f1(x2i)],  # 3
                    [x2j, -f1(x2j)],  # 4
                    [x2j, -f3(x2j, c3j * [1, -1], r3j)],  # 5
                    [x1j, -f3(x1j, c3j * [1, -1], r3j)],  # 6
                    [x1j, -f0(x1j)],
                    ],
            size=0.1,
            face_color='red',
            name=f'contacts-{i}-{j}',
            **extra_params,
            )

viewer.grid.enabled = True
viewer.grid.stride = -4

if __name__ == '__main__':
    napari.run()
