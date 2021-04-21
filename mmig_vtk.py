from numba.np.ufunc import parallel
import numpy as np
from scipy.spatial import cKDTree as KDTree # cKDTree is a C implementation of KDTree, which is faster
import open3d as o3d
import sys
from time import time
from vtk import vtkPolyDataReader, vtkDataReader, vtkArrayDataReader
from math import atan, cos, sin, pi
# import numba
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--global-base', '-g', type=float, help='Global interpolation baseline', default=0.0, required=False)
    parser.add_argument('--interpolation', '--inter', '-i', type=float, help='Total interpolation amount', required=True)
    parser.add_argument('--radius', '-r', type=float, help='Interpolation target radius', required=True)
    parser.add_argument('--mesh', '-m', type=str, help='Mesh file path', required=True)
    parser.add_argument('--vtk', '-v', type=str, help='VTK file path', required=True)

    args = parser.parse_args()

    start = time()
    _vtk = None
    vtk_mesh = None
    GLOB_BASE = args.global_base
    INTERPOLATION = args.interpolation
    _sphere = args.mesh
    # _ellipse = sys.argv[3]
    _vtk = args.vtk

    ANCHOR_POINT = np.array([0.0, 0.0])

    # GLOB_BASE = 0.6 # Global Baseline
    INT_STR = 1 - GLOB_BASE # Interpolation Strength

    print('loading meshes')
    _start = time()
    sphere = o3d.io.read_triangle_mesh(_sphere)
    # ellipse = o3d.io.read_triangle_mesh(_ellipse)
    if _vtk is not None:
        reader = vtkPolyDataReader()
        reader.SetFileName(_vtk)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.ReadAllColorScalarsOn()
        reader.Update()
        vtk_mesh = reader.GetOutput()
        wear = vtk_mesh.GetCellData().GetArray('wear')

    sphere.remove_duplicated_triangles()
    sphere.remove_duplicated_vertices()
    sphere.remove_degenerate_triangles()

    # ellipse.remove_duplicated_triangles()
    # ellipse.remove_duplicated_vertices()
    # ellipse.remove_degenerate_triangles()

    sphere_points = np.asarray(sphere.vertices)
    sphere_triangles = np.asarray(sphere.triangles)
    # ellipse_points = np.asarray(ellipse.vertices)
    _end = time()
    print(f'finished loading meshes in {_end - _start}')

    # mmig = mmig_class.mmig()

    print('Interpolating wear')
    _start = time()

    wear_total = {n: {'wear': 0.0, 'tally': 0} for n in range(sphere_points.shape[0])}
    max_vertex = np.array([args.radius, args.radius])
    max_scalar = 0
    np_wear = np.array(wear)
    avg_pnt = lambda p: np.mean((sphere_points[p[0]], sphere_points[p[1]], sphere_points[p[2]]), axis=0)
    avg_pnt_sphere = np.array(np.apply_along_axis(avg_pnt, 1, sphere_triangles))
    sphere_tree = KDTree(avg_pnt_sphere)
    query_result = sphere_tree.query_ball_point(avg_pnt_sphere, 0.15)
    smooth_fn = lambda near: np.mean([np_wear[idx] for idx in near])
    smooth_wear = np.apply_along_axis(np.vectorize(smooth_fn), 0, query_result)
    np_wear = smooth_wear

    for i, w in enumerate(np_wear):
        tri = sphere_triangles[i]
        for v in tri:
            wear_total[v]['wear'] += w
            wear_total[v]['tally'] += 1
    
    for key in wear_total:
        wear_total[key]['wear'] /= wear_total[key]['tally']
        max_scalar = max(max_scalar, wear_total[key]['wear'])

    for key in wear_total:
        wear_total[key]['wear'] /= max_scalar
    
    wear_total = np.array([wear_total[key]['wear'] for key in range(len(wear_total))])

    # @numba.njit('float64[:], float64', fastmath=True)
    def interpolate(s_p: np.ndarray, _wear: float):
        scale = (GLOB_BASE + _wear * INT_STR) * INTERPOLATION
        # Need to cos sin the sphere_point
        combined_sign = np.prod(np.sign(s_p[:2]))
        if s_p[1] < 0:
            slope_rad = atan(s_p[1] / s_p[0]) + pi # Assuming center is 0
        else:
            slope_rad = atan(s_p[1] / s_p[0])
        zsame_xymax = np.array([cos(slope_rad)*max_vertex[0]*combined_sign,
                                sin(slope_rad)*max_vertex[1]*combined_sign,
                                s_p[2]])
        s_p[...] = (1 - scale)*s_p + scale*zsame_xymax

    # @numba.njit('float64[:,:], float64[:], float64', parallel=True, fastmath=True)
    def run_interpolation(_sphere_points, _wear_total):
        for i in range(len(_wear_total)):
            interpolate(_sphere_points[i], _wear_total[i])

    # print(numba.typeof(sphere_points), numba.typeof(wear_total), numba.typeof(max_scalar))
    run_interpolation(sphere_points, wear_total)

    _end = time()
    print(f'Finished interpolating wear in {_end - _start}')

    if _sphere.split('.')[-1].lower() == "stl":
        sphere.compute_triangle_normals() # Required for STL
    o3d.io.write_triangle_mesh("interpolated."+_sphere.split('.')[-1], sphere)

    end = time()
    print(f'time taken was: {end-start}')
