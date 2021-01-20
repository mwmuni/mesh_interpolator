import stl
import numpy as np
from scipy.spatial import cKDTree as KDTree
import sys
from multiprocessing import Pool, cpu_count
import functools
from time import time

def process_kdtree_chunk(chunk, kdtree):
    return kdtree.query(chunk)

def hash_mesh(mesh, _verts, hash_dict):
    out_points = []
    for f in mesh:
        for v in _verts:
            v_hash = hash(f[v].tobytes())
            if v_hash not in hash_dict:
                hash_dict[v_hash] = []
                out_points.append(f[v])
    return out_points

def calc_dist(v, _sphere_points, _ellipse_points, _locations, _sphere_hash, _inter):
    v1 = _sphere_points[v]
    v2 = _ellipse_points[_locations[v]]
    alg = lambda i, j, k: (1 - k)*i + k*j
    new_loc = np.array([alg(v1[n], v2[n], _inter) for n in range(3)])
    _sphere_hash[hash(v1.tobytes())] = new_loc

def convert_hash(f, _sphere_hash):
    v0_h = _sphere_hash[hash(f[i_v0].tobytes())]
    v1_h = _sphere_hash[hash(f[i_v1].tobytes())]
    v2_h = _sphere_hash[hash(f[i_v2].tobytes())]

    for i in range(3):
        f[i_v0[i]] = v0_h[i]
        f[i_v1[i]] = v1_h[i]
        f[i_v2[i]] = v2_h[i]

sphere = []
ellipse = []
sphere_points = []
ellipse_points = []
sphere_hash = {}
ellipse_hash = {}

if __name__ == '__main__':
    start = time()
    if len(sys.argv) > 1:
        _INTERPOLATION = float(sys.argv[1])
        _sphere = sys.argv[2]
        _ellipse = sys.argv[3]
    else:
        _sphere = "sphere.stl"
        _ellipse = "ellipse.stl"
        _INTERPOLATION = 0.5

    print('loading meshes')
    with Pool(2) as pool:
        threads = [pool.apply_async(stl.Mesh.from_file, (_sphere,)),
        pool.apply_async(stl.Mesh.from_file, (_ellipse,))]
        sphere, ellipse = [p.get() for p in threads]
    print('finished loading meshes')

    INTERPOLATION = _INTERPOLATION

    nn_pairs = {}

    i_v0 = [0,1,2]
    i_v1 = [3,4,5]
    i_v2 = [6,7,8]

    verts = [i_v0, i_v1, i_v2]

    print('hashing meshes')
    with Pool(2) as pool:
        threads = [pool.apply_async(hash_mesh, (sphere, verts, sphere_hash)),
            pool.apply_async(hash_mesh, (ellipse, verts, ellipse_hash))]
        sphere_points, ellipse_points = [p.get() for p in threads]
    print('finished hashing meshes')

    kdtree = KDTree(ellipse_points, leafsize=90)

    print('about to query')
    with Pool(cpu_count()) as p:
        locations = p.map(functools.partial(process_kdtree_chunk, kdtree=kdtree), sphere_points, len(sphere_points)//cpu_count())
    print('finised query')

    locations = [l[1] for l in locations]

    for v in range(len(locations)):
        v1 = sphere_points[v]
        v2 = ellipse_points[locations[v]]
        new_loc = np.array([(1 - INTERPOLATION)*v1[0] + INTERPOLATION*v2[0],
                            (1 - INTERPOLATION)*v1[1] + INTERPOLATION*v2[1],
                            (1 - INTERPOLATION)*v1[2] + INTERPOLATION*v2[2]])
        sphere_hash[hash(v1.tobytes())] = new_loc

    for f in sphere:
        v0_h = sphere_hash[hash(f[i_v0].tobytes())]
        v1_h = sphere_hash[hash(f[i_v1].tobytes())]
        v2_h = sphere_hash[hash(f[i_v2].tobytes())]

        for i in range(3):
            f[i_v0[i]] = v0_h[i]
            f[i_v1[i]] = v1_h[i]
            f[i_v2[i]] = v2_h[i]

    sphere.save('interpolated.stl')
    end = time()
    print(f'time taken was: {end-start}')
