# mesh_interpolator

REQUIREMENTS: numpy, numpy-stl, scipy
OPTIONAL: Open3D

An algorithm that will interpolate between two given mesh files.

Rename the 'sphere.stl' and 'ellipse.stl' in the code to the meshes you wish to use. Change the value of 'INTERPOLATION' to choose how far the mesh will be interpolated. (between 0 and 1)
Or simply run it with the arguments:
python mmig.py \[ratio] \[source_file.stl] \[target_file.stl]

Example:
python mmig.py 0.5 sphere.stl ellipse.stl

The mmig_open3d.py version leverages Open3D to speed up the interpolation drastically (around 5x speedup). The Open3D version is strongly recommended over the numpy-stl version not only for speed, but for the ability to use file types other than STL.

SUPPORTED FILE TYPES (mmig_open3d.py only): STL, PLY, OFF, OBJ, GLTF

Only STL seems to store the vertex/triangle normals, so it may be best to stick to STL files.
