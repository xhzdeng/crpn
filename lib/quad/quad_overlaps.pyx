
import numpy as np
cimport numpy as np

cdef extern from 'QuadOverlaps.hpp':
    float QuadOverlaps(np.float32_t* a, np.float32_t* b)

def quad_overlaps(np.ndarray[np.float32_t, ndim=2] quads,
                  np.ndarray[np.float32_t, ndim=2] query_quads):
    cdef unsigned int N = quads.shape[0]
    cdef unsigned int K = query_quads.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] overlaps = np.zeros((N, K), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] a = np.zeros(8, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] b = np.zeros(8, dtype=np.float32)
    cdef unsigned int k, n
    for k in range(K):
        a = query_quads[k, :]
        for n in range(N):
            b = quads[n, :]
            overlaps[n, k] = QuadOverlaps(&a[0], &b[0])
    return overlaps

