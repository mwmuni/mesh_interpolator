import stl
import numpy as np
from scipy.spatial import KDTree
import sys

if len(sys.argv) > 1:
    _INTERPOLATION = float(sys.argv[1])
    _sphere = sys.argv[2]
    _ellipse = sys.argv[3]
else:
    _sphere = "sphere.stl"
    _ellipse = "ellipse.stl"
    _INTERPOLATION = 0.5

sphere = stl.Mesh.from_file(_sphere)
ellipse = stl.Mesh.from_file(_ellipse)

INTERPOLATION = _INTERPOLATION

sphere_hash = {}
ellipse_hash = {}

nn_pairs = {}

i_v0 = [0,1,2]
i_v1 = [3,4,5]
i_v2 = [6,7,8]

verts = [i_v0, i_v1, i_v2]

sphere_points = []

for f in sphere:
    for v in verts:
        v_hash = hash(f[v].tobytes())
        if v_hash not in sphere_hash:
            sphere_hash[v_hash] = []
            sphere_points.append(f[v])

ellipse_points = []

for f in ellipse:
    for v in verts:
        v_hash = hash(f[v].tobytes())
        if v_hash not in ellipse_hash:
            ellipse_hash[v_hash] = []
            ellipse_points.append(f[v])

kdtree = KDTree(ellipse_points)

distances, locations = kdtree.query(sphere_points)

for v in range(len(locations)):
    v1 = sphere_points[v]
    v2 = ellipse_points[locations[v]]
    alg = lambda i, j, k: (1 - k)*i + k*j
    new_loc = np.array([alg(v1[n], v2[n], INTERPOLATION) for n in range(3)])
    sphere_hash[hash(v1.tobytes())] = new_loc

count = 0

for f in sphere:
    v0_h = sphere_hash[hash(f[i_v0].tobytes())]
    v1_h = sphere_hash[hash(f[i_v1].tobytes())]
    v2_h = sphere_hash[hash(f[i_v2].tobytes())]

    for i in range(3):
        f[i_v0[i]] = v0_h[i]
        f[i_v1[i]] = v1_h[i]
        f[i_v2[i]] = v2_h[i]
    count += 1

sphere.save('interpolated.stl')
# sphere.save('te.stl')
