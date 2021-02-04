import numpy as np
from scipy.spatial import cKDTree as KDTree # cKDTree is a C implementation of KDTree, which is faster
import open3d as o3d
import sys
from multiprocessing import Pool, cpu_count # Multiprocessing allows for running on multiple cores
import functools # Used for the 'partial' function, which allows assigning keyword variables to multithreaded function mapping
from time import time
import mmig_class

if __name__ == '__main__':
    start = time()
    if len(sys.argv) > 1:
        INTERPOLATION = float(sys.argv[1])
        _sphere = sys.argv[2]
        _ellipse = sys.argv[3]
    else:
        _sphere = "sphere.stl"
        _ellipse = "ellipse.stl"
        INTERPOLATION = 0.5

    print('loading meshes')
    _start = time()
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

    kdtree = KDTree(ellipse_points, leafsize=90)
    mmig = mmig_class.mmig()

    print('about to query')
    _start = time()
    with Pool(cpu_count()) as p:
        locations = p.map(functools.partial(mmig.process_kdtree_chunk, kdtree=kdtree), sphere_points, sphere_points.shape[0]//cpu_count())
    _end = time()
    print(f'finised query in {_end - _start}')

    locations = [l[1] for l in locations]

    print('about to interpolate')
    _start = time()
    # lam = lambda x: (1 - INTERPOLATION)*x[0] + INTERPOLATION*x[1]
    # lam = lambda x: x[0]+(-(x[0]-x[1])*INTERPOLATION)
    # inter = np.array(np.apply_along_axis(lam, 1, np.array([(s_p[1], ellipse_points[locations[s_p[0][0]]][s_p[0][1]]) for s_p in np.ndenumerate(sphere_points)], dtype=object)))
    # inter = np.array(np.apply_along_axis(lam, 1, np.array([(s_p[1], ellipse_points[locations[s_p[0][0]]][s_p[0][1]]) for s_p in np.ndenumerate(sphere_points)], dtype=object)))
    # sphere_points[:] = inter.reshape((-1,3))
    # For some reason, using lambda increases processing time by 0.7s, so I'll stick with partial
    inter = np.array(np.apply_along_axis(functools.partial(mmig.interpolate, interpolation=INTERPOLATION), 1, np.array([(s_p[1], ellipse_points[locations[s_p[0][0]]][s_p[0][1]]) for s_p in np.ndenumerate(sphere_points)], dtype=object)))
    sphere_points[:] = inter[:, 0].reshape((-1,3))
    _end = time()
    print(f'finised interpolating in {_end - _start}')

    if _sphere.split('.')[-1].lower() == "stl":
        sphere.compute_triangle_normals() # Required for STL
    o3d.io.write_triangle_mesh("interpolated."+_sphere.split('.')[-1], sphere)

    end = time()
    print(f'time taken was: {end-start}')
