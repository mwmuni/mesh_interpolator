import numpy as np
from scipy.spatial import cKDTree as KDTree
import open3d as o3d
import sys
from multiprocessing import Pool, cpu_count
import functools
from time import time

def process_kdtree_chunk(chunk, kdtree):
    return kdtree.query(chunk)

def interpolate(s_p):
        s_p[0] = (1 - INTERPOLATION)*s_p[0] + INTERPOLATION*s_p[1]
        return s_p

INTERPOLATION = 0.5

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

    _start = time()
    print('loading meshes')
    sphere = o3d.io.read_triangle_mesh(_sphere)
    ellipse = o3d.io.read_triangle_mesh(_ellipse)

    sphere.remove_duplicated_triangles()
    sphere.remove_duplicated_vertices()
    sphere.remove_degenerate_triangles()

    ellipse.remove_duplicated_triangles()
    ellipse.remove_duplicated_vertices()
    ellipse.remove_degenerate_triangles()

    sphere_points = np.asarray(sphere.vertices)
    ellipse_points = np.asarray(ellipse.vertices)
    _end = time()
    print(f'finished loading meshes in {_end - _start}')

    INTERPOLATION = _INTERPOLATION

    kdtree = KDTree(ellipse_points, leafsize=90)

    _start = time()
    print('about to query')
    with Pool(cpu_count()) as p:
        locations = p.map(functools.partial(process_kdtree_chunk, kdtree=kdtree), sphere_points, sphere_points.shape[0]//cpu_count())
    _end = time()
    print(f'finised query in {_end - _start}')

    locations = [l[1] for l in locations]

    _start = time()
    print('about to interpolate')
    inter = np.array(np.apply_along_axis(interpolate, 1, np.array([(s_p[1], ellipse_points[locations[s_p[0][0]]][s_p[0][1]]) for s_p in np.ndenumerate(sphere_points)], dtype=object)))
    sphere_points[:] = inter[:, 0].reshape((-1,3))
    _end = time()
    print(f'finised interpolating in {_end - _start}')

    sphere.compute_triangle_normals()
    o3d.io.write_triangle_mesh("interpolated.stl", sphere)

    end = time()
    print(f'time taken was: {end-start}')
