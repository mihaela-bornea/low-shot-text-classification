import scipy
import heapq
from scipy.sparse import csr_matrix
import numpy as np


def pointwise_mult(a1, a2):
    if isinstance(a1, csr_matrix):
        m = a1.multiply(a2)
        return m.tocsr()
    else:
        return a1 * a2

def mat_average(mat, axis=0):
    if isinstance(mat, csr_matrix):
        avg_mat = np.asarray(mat.mean(axis=axis))
    else:
        avg_mat = np.average(mat, axis=axis)
        avg_mat = np.reshape(avg_mat, (1,) + avg_mat.shape)
    return avg_mat

def mat_sum(mat, axis=0):
    if isinstance(mat, csr_matrix):
        sum_mat = np.asarray(mat.sum(axis=axis))
    else:
        sum_mat = np.sum(mat, axis=axis)
        sum_mat = np.reshape(sum_mat, (1,) + sum_mat.shape)
    return sum_mat

def matmul(mat1, mat2):
    if isinstance(mat1, csr_matrix):
        m = mat1*mat2
        return m
    else:
        return np.matmul(mat1, mat2)

def compute_norm(vecs):
    norms = np.sqrt(np.sum(pointwise_mult(vecs, vecs), axis=1))
    norms[norms == 0] = 1
    norms = norms.reshape((norms.shape[0],1))

    return norms

def mat_concat(matrices):
    if isinstance(matrices[0], csr_matrix):
        return scipy.sparse.vstack(matrices).tocsr()
    else:
        return np.concatenate(matrices)