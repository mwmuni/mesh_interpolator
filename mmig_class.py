import numpy as np
from scipy.spatial import cKDTree as KDTree
from typing import Tuple

class mmig:
    def __init__(self):
        pass

    @staticmethod
    def process_kdtree_chunk(chunk: np.array, kdtree: KDTree):
        """Run KDTree on a chunk of data
        
        Allows for splitting up the KDTree query to multiple cores, drastically improving speed"""
        return kdtree.query(chunk)

    @staticmethod
    def interpolate(s_p: Tuple[float, float], interpolation: float) -> Tuple[float, float]:
        """Moves s_p[0] towards s_p[1] by a factor of INTERPOLATION
        
        Used for np.apply_along_axis, this function gets applied vertically along the input array"""
        s_p[0] = (1 - interpolation)*s_p[0] + interpolation*s_p[1]
        # s_p[0] = s_p[0]+(-(s_p[0]-s_p[1])*interpolation)
        return s_p
    
    @staticmethod
    def interpolate_xy(s_p: Tuple[float, float], interpolation: float) -> Tuple[float, float]:
        """Moves s_p[0] towards s_p[1] by a factor of INTERPOLATION leaving the z axis for s_p[0] unchanged
        
        Used for np.apply_along_axis, this function gets applied vertically along the input array"""
        s_p[0] = np.array([*((1 - interpolation)*s_p[0, :2] + interpolation*s_p[1]), s_p[0, 2]])
        # s_p[0] = s_p[0]+(-(s_p[0]-s_p[1])*interpolation)
        return s_p