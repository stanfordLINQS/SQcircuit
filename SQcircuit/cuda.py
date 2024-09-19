from typing import Tuple

from scipy.sparse import csr_matrix as csr_cpu

try:
    from cupy import ndarray as ndarray_gpu
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
    from cupyx.scipy.sparse.linalg import eigsh as eigsh_gpu


    def _diag_sparse_gpu(
        A: csr_gpu,
        n_eig: int
    ) -> Tuple[ndarray_gpu]:
        efreqs, evecs = eigsh_gpu(
            A, k=n_eig, which='SA'
        )

        return efreqs, evecs

    def _csr_to_gpu(
        A: csr_cpu
    ) -> csr_gpu:
        return csr_gpu(A)

except ModuleNotFoundError:
    pass

# TODO: support devices
def diag_sparse_gpu(A: csr_cpu , n_eig: int):
    try:
        return _diag_sparse_gpu(_csr_to_gpu(A), n_eig)
    except NameError:
        raise ModuleNotFoundError('CuPy is not available on this machine; '
                                  'you cannot run `diag_sparse_gpu()`. ')
