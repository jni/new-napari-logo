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
c0 = (0, 0)
r1 = 1
c1 = (phi, 0)
r3s = [phi - 1, 1, phi]
c3s = np.array([
        [4*phi - 6, 2 * np.sqrt(10*phi - 3 * phi**2 - 8)],
        [2 - phi, 2 * np.sqrt(phi - 1)],
        [2*phi - 3, 2 * np.sqrt(2*phi - 2)],
        ])

x1s = r0 * c3s[:, 0] / c3s[:, 1]
x2s = phi + (c3s[:, 0] - phi) / c3s[:, 1]

domain = np.array([1 - phi, phi + 1])
xs = np.linspace(*domain, num=500, endpoint=True)

lines_top = []
lines_bottom = []

for r3, c3, x1, x2 in zip(r3s, c3s, x1s, x2s):
    xs0 = xs[xs < x1]
    xs1 = xs[(x1 <= xs) & (xs < x2)]
    xs2 = xs[x2 <= xs]

    ys0 = np.sqrt(np.clip(r0**2 - xs0**2, 0, None))
    ys1 = c3[1] + np.sqrt(r3**2 - (xs1 - c3[0])**2)
    ys2 = np.sqrt(np.clip(r1**2 - (xs2 - c1[0])**2, 0, None))

    ys_top = np.concatenate([ys0, ys1, ys2])
    line_top = np.stack([xs, ys_top], axis=1)
    lines_top.append(line_top)

    line_bottom = np.copy(line_top[-2:0:-1])
    line_bottom[:, 1] *= -1

    lines_bottom.append(line_bottom)

candidate_shapes = [
        np.concatenate((line_top, line_bottom), axis=0) for line_top,
        line_bottom in itertools.product(lines_top, lines_bottom)
        ]

viewer = napari.Viewer()
for i, shp in enumerate(candidate_shapes):
    viewer.add_shapes(
            [shp, shp],
            shape_type='polygon',
            scale=[500, 500],
            edge_width=[2, 1],
            edge_color=[SAND, FOREST],
            face_color=[LAGOON, LAGOON],
            name=f'option-{i}',
            )

viewer.grid.enabled = True

if __name__ == '__main__':
    napari.run()
