# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        thread_index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if thread_index < out_size:
            to_index(thread_index, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            out[thread_index] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore



def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)
            out[i] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    THREADS_PER_BLOCK = 32

    shared_memory = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
    global_index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_index = cuda.threadIdx.x

    if global_index < size:
        shared_memory[local_index] = a[global_index]
    else:
        shared_memory[local_index] = 0.0

    cuda.syncthreads()

    step = 1
    while step < THREADS_PER_BLOCK:
        if local_index % (2 * step) == 0:
            shared_memory[local_index] += shared_memory[local_index + step]
        step *= 2
        cuda.syncthreads()

    if local_index == 0:
        out[cuda.blockIdx.x] = shared_memory[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        THREADS_PER_BLOCK = 1024
        shared_mem = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)
        current_out_index = cuda.local.array(MAX_DIMS, numba.int32)
        block_index = cuda.blockIdx.x
        thread_index = cuda.threadIdx.x

        if block_index < out_size:
            shared_mem[thread_index] = reduce_value
            to_index(block_index, out_shape, current_out_index)
            current_out_index[reduce_dim] = current_out_index[reduce_dim] * THREADS_PER_BLOCK + thread_index

            if current_out_index[reduce_dim] < a_shape[reduce_dim]:
                shared_mem[thread_index] = a_storage[index_to_position(current_out_index, a_strides)]
                cuda.syncthreads()

                step = 1
                while step < THREADS_PER_BLOCK:
                    if thread_index % (step * 2) == 0 and thread_index + step < THREADS_PER_BLOCK:
                        shared_mem[thread_index] = fn(shared_mem[thread_index], shared_mem[thread_index + step])
                    cuda.syncthreads()
                    step *= 2

            if thread_index == 0:
                out[block_index] = shared_mem[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    THREADS_PER_BLOCK = 32
    row = cuda.threadIdx.y
    col = cuda.threadIdx.x
    shared_a = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    shared_b = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)

    if row < size and col < size:
        shared_a[row, col] = a[row * size + col]
        shared_b[row, col] = b[row * size + col]

    cuda.syncthreads()
    if row < size and col < size:
        partial_sum = 0.0
        for k in range(size):
            partial_sum += shared_a[row, k] * shared_b[k, col]

        out[row * size + col] = partial_sum


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    global_row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    global_col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    local_row = cuda.threadIdx.x
    local_col = cuda.threadIdx.y

    result = 0.0
    num_a_rows = a_shape[-2]
    num_shared_dim = a_shape[-1]
    num_b_cols = b_shape[-1]

    for block_offset in range(0, num_shared_dim, BLOCK_DIM):
        a_row = global_row
        a_col = block_offset + local_row
        if a_row < num_a_rows and a_col < num_shared_dim:
            shared_a[local_row, local_col] = a_storage[
                batch * a_batch_stride + a_row * a_strides[-2] + a_col * a_strides[-1]
            ]
        else:
            shared_a[local_row, local_col] = 0.0

        b_row = block_offset + local_col
        b_col = global_col
        if b_row < num_shared_dim and b_col < num_b_cols:
            shared_b[local_row, local_col] = b_storage[
                batch * b_batch_stride + b_row * b_strides[-2] + b_col * b_strides[-1]
            ]
        else:
            shared_b[local_row, local_col] = 0.0

        cuda.syncthreads()

        for k in range(min(BLOCK_DIM, num_shared_dim - block_offset)):
            result += shared_a[local_row, k] * shared_b[k, local_col]

        cuda.syncthreads()

    if global_row < num_a_rows and global_col < num_b_cols:
        out[
            batch * out_strides[0] + global_row * out_strides[1] + global_col * out_strides[2]
        ] = result


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
