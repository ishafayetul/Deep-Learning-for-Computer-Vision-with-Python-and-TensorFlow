# 📘 TensorFlow Code Summary

## 🧱 Tensor Creation & Initialization
- **tf.constant()**: Creates fixed tensors (supports multi-dimensional: 3D, 4D).
- **tf.convert_to_tensor(np_array)**: Converts NumPy array to Tensor.
- **tf.eye()**: Creates identity matrices, supports batch identity tensors.
- **tf.fill(shape, value)**: Fills a tensor with a constant value.
- **tf.ones(), tf.zeros()**: Create tensors filled with 1s or 0s.
- **tf.size()**: Returns number of elements in tensor.

## 🎲 Random Tensor Generation
- **tf.random.normal()**: Tensor with normally distributed values.
- **tf.random.uniform()**: Tensor with uniformly distributed values.

## ✂️ Tensor Slicing & Indexing
- Standard Python-style slicing (`tensor[:2, 1:3]`) is used to extract rows/columns or submatrices.

## ➕ Tensor Operations
- **tf.math.argmax() / tf.math.argmin()**: Index of max/min along a specific axis.
- **tf.math.pow()**: Element-wise exponentiation.
- **tf.math.reduce_sum()**: Sum across dimensions.
- **tf.math.sigmoid()**: Applies sigmoid activation function element-wise.

## 🔁 Reshaping & Manipulation
- **tf.reshape(tensor, shape)**: Changes the shape of tensor.
- **tf.expand_dims(tensor, axis)**: Adds new dimensions.
- **tf.squeeze(tensor)**: Removes size-1 dimensions.
- **tf.concat([t1, t2], axis=n)**: Concatenates tensors along given axis.
- **tf.stack([t1, t2], axis=n)**: Stacks tensors along new axis.

## 🧮 Linear Algebra Ops
- **tf.linalg.matmul(t1, t2) / t1 @ t2**: Matrix multiplication.
- **tf.linalg.trace()**: Sum of diagonal elements.
- **tf.linalg.transpose() / tf.linalg.matrix_transpose()**: Transposes a tensor.
- **tf.linalg.inv()**: Inverse of a matrix.
- **tf.linalg.det()**: Determinant of a matrix.
- **tf.linalg.adjoint()**: Conjugate transpose (adjoint) of matrix.
- **tf.linalg.cross(t1, t2)**: Element-wise 3D cross product.

## 📉 Decompositions
- **Cholesky (tf.linalg.cholesky())**: Lower-triangular decomposition for symmetric matrices.
- **QR Decomposition (tf.linalg.qr())**: Decomposes tensor into orthogonal Q and upper-triangular R.
- **SVD (tf.linalg.svd())**: Singular Value Decomposition returns singular values and U/V matrices.
- **Eigen (tf.linalg.eigh())**: Computes eigenvalues and eigenvectors for symmetric matrices.

## 🧩 Advanced Tensor Ops
- **tf.linalg.band_part(tensor, num_lower, num_upper)**: Extracts band part of a tensor (diagonal + nearby).
- **tf.pad(tensor, paddings, mode)**: Pads tensor with specified values and mode (CONSTANT, REFLECT, etc.).

## 🔁 Einstein Summation (tf.einsum)
A flexible, compact way to define operations:
- `"ij,jk->ik"`: Matrix multiplication
- `"ij->ji"`: Transpose
- `"bij->"`: Sum all elements
- `"bij->bj"`: Sum columns
- `"bij,bjk,bkl->bil"`: Batch matrix multiplication

Used for complex reductions, summations, and contractions.

## 🎯 tf.gather
- **tf.gather(tensor, indices, axis)**: Selects rows or columns by index.  
  Useful for slicing specific axes.

## 🔍 tf.gather_nd
- **tf.gather_nd(tensor, indices)**: Selects values based on multi-dimensional indices.  
  Deep indexing: `t[1,1,1]`, `t[0,2,2]`, etc.  
  `batch_dims=n` allows partial indexing per batch.

