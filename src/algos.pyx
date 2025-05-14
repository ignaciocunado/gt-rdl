# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy
from libc.stdint cimport int16_t

def floyd_warshall(adjacency_matrix):
    cdef unsigned int n = adjacency_matrix.shape[0]
    assert adjacency_matrix.shape[1] == n

    # Copy into a C-contiguous int16 array
    cdef numpy.ndarray[numpy.int16_t, ndim=2, mode='c'] M = \
        adjacency_matrix.astype(numpy.int16, order='C', casting='safe', copy=True)
    cdef numpy.ndarray[numpy.int16_t, ndim=2, mode='c'] path = \
        -1 * numpy.ones((n, n), dtype=numpy.int16)

    cdef unsigned int i, j, k
    cdef int16_t M_ij, M_ik, cost_ikkj
    cdef int16_t* M_ptr = <int16_t*> &M[0, 0]
    cdef int16_t* M_i_ptr
    cdef int16_t* M_k_ptr

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # Floyd–Warshall core
    for k in range(n):
        M_k_ptr = M_ptr + n * k
        for i in range(n):
            M_i_ptr = M_ptr + n * i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510 | M[i][j] <= -510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path


def get_all_edges(path, i, j):
    cdef int k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)


def gen_edge_input(max_dist, path, edge_feat):

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(numpy.int64, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(numpy.int64, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all
