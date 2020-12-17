# mesh_interpolator

REQUIREMENTS: numpy, numpy-stl, scipy

An algorithm that will interpolate between two given stl files.

Rename the 'sphere.stl' and 'ellipse.stl' in the code to the meshes you wish to use.

Change the value of 'INTERPOLATION' to choose how far the mesh will be interpolated. (between 0 and 1)

Can use command line arguments to set these values in the order:
  python mmig.py 0.5 sphere.stl ellipse.stl
