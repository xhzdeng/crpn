
import numpy as np
cimport numpy as np

cdef extern from 'SortPoints.hpp':
    void SortPoints(np.float32_t*)
    void SortPoints(np.float32_t*, np.float32_t*)

def sort_points(np.ndarray[np.float32_t, ndim=2] quads):
    cdef int nquads = quads.shape[0]
    if nquads == 0:
        return quads
    cdef np.ndarray[np.float32_t, ndim=1] qd = np.zeros(8, dtype=np.float32)
    cdef int _i
    for _i in range(nquads):
        qd = quads[_i, :]
        SortPoints(&qd[0])
    return quads
