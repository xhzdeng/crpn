import numpy as np
cimport numpy as np

cdef extern from 'Quad2Obb.hpp':
    void Quad2Obb(np.float32_t*, np.float32_t*)

def quad_2_obb(np.ndarray[np.float32_t, ndim=2] quads):
    cdef int nquads = quads.shape[0]
    if nquads == 0:
        return quads
    cdef np.ndarray[np.float32_t, ndim=1] qd = np.zeros(8, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] tmp = np.zeros(5, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] tmps = np.zeros((nquads, 5), dtype=np.float32)
    cdef int _i
    for _i in range(nquads):
        qd = quads[_i, :]
        tmp = tmps[_i, :]        
        Quad2Obb(&qd[0], &tmp[0])
    return tmps
