import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
import numpy as np

# ======================= 1. 3D Tensor ========================
print("\n===== 1. 3D Tensor (shape: 3x2x3) =====")
threeDtensor = tf.constant([
    [[1, 2, 3], [7, 8, 9]],
    [[7, 5, 3], [4, 3, 2]],
    [[1, 5, 6], [7, 5, 4]]
])
print("3D Tensor:\n", threeDtensor)

# ======================= 2. 4D Tensor ========================
print("\n===== 2. 4D Tensor (shape: 2x2x2x3) =====")
fourDtensor = tf.constant([
    [[[1, 2, 3], [3, 5, 9]],
     [[1, 5, 9], [7, 3, 5]]],

    [[[1, 5, 7], [7, 8, 6]],
     [[7, 5, 6], [1, 2, 4]]]
])
print("4D Tensor:\n", fourDtensor)
print("Rank (ndim) of 4D Tensor:", fourDtensor.ndim)

# ======================= 3. NumPy to Tensor ========================
print("\n===== 3. Convert NumPy to Tensor =====")
arr = np.array([1, 2, 3])
converted_tensor = tf.convert_to_tensor(arr)
print("NumPy Array:\n", arr)
print("Converted Tensor:\n", converted_tensor)

# ======================= 4. Identity Matrix ========================
print("\n===== 4. Identity Matrix (batch_shape=[2], shape: 2x5x5) =====")
eye_tensor = tf.eye(
    num_rows=5,
    num_columns=5,
    batch_shape=[2]
)
print("Identity Tensor:\n", eye_tensor)

# ======================= 5. Fill / Ones / Zeros / Size ========================
print("\n===== 5. Fill, Ones, Zeros, and Size =====")
filled_tensor = tf.fill([2, 3], 5)
ones_tensor = tf.ones([2, 3])
zero_tensor = tf.zeros([2, 3])

print("Filled Tensor (2x3 with 5):\n", filled_tensor)
print("Ones Tensor (2x3):\n", ones_tensor)
print("Zeros Tensor (2x3):\n", zero_tensor)

tensor = tf.constant([
    [1, 2, 3],
    [3, 5, 6]
])
size = tf.size(tensor)
print("Size of Tensor (number of elements):", size.numpy())

# ======================= 6. Random Tensors ========================
print("\n===== 6. Random Tensors =====")
normal_random_tensor = tf.random.normal(
    shape=[2, 3],
    mean=100.0,
    stddev=5.0,
    dtype=tf.dtypes.float16
)
print("Normal Random Tensor (mean=100, stddev=5):\n", normal_random_tensor)

uniform_random_tensor = tf.random.uniform(
    shape=[2, 3],
    minval=2,
    maxval=100,
    dtype=tf.dtypes.int32,
    seed=10
)
print("Uniform Random Tensor (min=2, max=100):\n", uniform_random_tensor)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import tensorflow as tf

# ===================== Base Tensor =====================
tensor = tf.constant([
    [1, 2, 3],
    [5, 6, 7],
    [8, 9, 10]
])

print("Original Tensor (3x3):\n", tensor)

# ===================== Tensor Slicing =====================

print("\n1. tensor[:,:] => All rows and columns (entire tensor):")
print(tensor[:, :])

print("\n2. tensor[0,:] => First row:")
print(tensor[0, :])

print("\n3. tensor[:,0] => First column:")
print(tensor[:, 0])

print("\n4. tensor[:1,:1] => First element (top-left):")
print(tensor[:1, :1])

print("\n5. tensor[:2,:3] => First 2 rows and all 3 columns:")
print(tensor[:2, :3])

print("\n6. tensor[1:2,:3] => Only 2nd row (row index 1):")
print(tensor[1:2, :3])

print("\n7. tensor[:3,1:2] => All rows, only 2nd column:")
print(tensor[:3, 1:2])

print("\n8. tensor[2:3,2:3] => Last element (bottom-right corner):")
print(tensor[2:3, 2:3])

print("\n9. tensor[0:-2,0:-2] => From row 0 to second-last, and col 0 to second-last:")
print(tensor[0:-2, 0:-2])
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow debug logs
import tensorflow as tf

# ========================================================
# 📌 TensorFlow Math Functions - Notes & Examples
# ========================================================

# ============================ tf.math.argmax ============================
print("==== tf.math.argmax ====")
'''
🔹 tf.math.argmax(<tensor>, axis={0/1})
   Returns the index of the maximum value along a specified axis.
'''
tensor = tf.constant([[10, 20], [30, 5]])
print("Tensor:\n", tensor.numpy())

print("Argmax axis=0 (column-wise):", tf.math.argmax(tensor, axis=0).numpy())  # ➝ [1 0]
print("Argmax axis=1 (row-wise):", tf.math.argmax(tensor, axis=1).numpy())    # ➝ [1 0]


# ============================ tf.math.argmin ============================
print("\n==== tf.math.argmin ====")
'''
🔹 tf.math.argmin(<tensor>, axis={0/1})
   Returns the index of the minimum value along a specified axis.
'''
print("Argmin axis=0 (column-wise):", tf.math.argmin(tensor, axis=0).numpy())  # ➝ [0 1]
print("Argmin axis=1 (row-wise):", tf.math.argmin(tensor, axis=1).numpy())    # ➝ [0 1]


# ============================ tf.math.pow ============================
print("\n==== tf.math.pow ====")
'''
🔹 tf.math.pow(tensor1, tensor2)
   Element-wise exponentiation: tensor1^tensor2
'''
a = tf.constant([2, 3], dtype=tf.float32)
b = tf.constant([3, 2], dtype=tf.float32)

print("a:", a.numpy())               # ➝ [2.0, 3.0]
print("b:", b.numpy())               # ➝ [3.0, 2.0]
print("a^b:", tf.math.pow(a, b).numpy())  # ➝ [8.0 9.0]


# ============================ tf.math.reduce_sum ============================
print("\n==== tf.math.reduce_sum ====")
'''
🔹 tf.math.reduce_sum(tensor, axis=None)
   Sums all elements in the tensor or across specific dimensions.
'''
tensor_sum = tf.constant([[1, 2], [3, 4]])
print("Tensor:\n", tensor_sum.numpy())

print("Sum of all elements:", tf.math.reduce_sum(tensor_sum).numpy())               # ➝ 10
print("Sum along axis=0 (columns):", tf.math.reduce_sum(tensor_sum, axis=0).numpy())  # ➝ [4 6]
print("Sum along axis=1 (rows):", tf.math.reduce_sum(tensor_sum, axis=1).numpy())    # ➝ [3 7]


# ============================ tf.math.sigmoid ============================
print("\n==== tf.math.sigmoid ====")
'''
🔹 tf.math.sigmoid(tensor)
   Applies the sigmoid activation function: 1 / (1 + e^(-x))
'''
x = tf.constant([-1.0, 0.0, 1.0])
print("Input:", x.numpy())                 # ➝ [-1.0, 0.0, 1.0]
print("Sigmoid:", tf.math.sigmoid(x).numpy())  # ➝ [0.2689 0.5 0.7311]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Base tensors for operations
t1 = tf.constant([[1, 2, 3], [4, 7, 6]])
t2 = tf.constant([[7, 8, 5], [8, 9, 3], [4, 5, 1]])
t3 = tf.constant([[1, 3, 5], [2, 4, 6], [3, 6, 9]])

# ---------------- Adjoints and Inverses ----------------
adjoint = tf.linalg.adjoint(t2)
print("Adjoint of t2:", adjoint)

print("Inverse of t2:", tf.linalg.inv(tf.cast(t2, tf.float32)))
print("Determinant of t2:", tf.linalg.det(tf.cast(t2, tf.float32)))
print("==========================================")

# ---------------- Matrix Operations ----------------
t1xt2 = tf.linalg.matmul(t1, t2)
print("Matrix multiplication (t1 x t2):", t1xt2)
print("Matrix multiplication using @ operator:", t1 @ t2)

print("Trace of t1:", tf.linalg.trace(t1))
print("Determinant of t2:", tf.linalg.det(tf.cast(t2, tf.float32)))
print("Inverse of t2:", tf.linalg.inv(tf.cast(t2, tf.float32)))
print("Transpose of t1:", tf.linalg.matrix_transpose(t1))
print("Element-wise cross product of t2 and t3:", tf.linalg.cross(t2, t3))
print("======================================================")

# ---------------- Cholesky Decomposition ----------------
'''Cholesky Decomposition: A = L * L^T where L is lower triangular
matrix must be sq and symmetric'''
sym_tensor = tf.constant([[4, 2], [2, 3]])
cholesky_decomp = tf.linalg.cholesky(tf.cast(sym_tensor, tf.float32))
print("Cholesky decomposition of sym_tensor:", cholesky_decomp)
print("Reconstructed Matrix (L @ L^T):", cholesky_decomp @ tf.transpose(cholesky_decomp))
print("======================================================")

# ---------------- QR Decomposition ----------------
'''QR Decomposition: A = Q * R where Q is orthogonal, R is upper triangular
*Orthogonal Mat: [A] is an orthogonal mat if [A]^T=[A]^-1 
* return two tensors Q & R
* [A] must not be int'''
tensor = tf.constant([[1.0, 2.0, 3.0], [3.0, 5.0, 6.0], [4.0, 7.0, -9.0]])
q, r = tf.linalg.qr(tensor, full_matrices=True)
print("Q matrix:", q)
print("R matrix:", r)
print("Reconstructed A from Q and R:", q @ r)
print("======================================================")

# ---------------- Singular Value Decomposition (SVD) ----------------
'''SVD: A = U * S * V^T
*S= MxN Diagonal Matrix ->  its diagonal values are singular values of A
*U= MxM orthogonal matrix -> its columns are left singular vectors of A
*V= NxN orthogonal Matrix ->  its columns are right singular vectors of A'''
mat = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
s, u, v = tf.linalg.svd(mat, full_matrices=True)
print("Singular values:", s)
print("Left singular vectors (U):", u)
print("Right singular vectors (V^T):", v)
print("==============================================")

# ---------------- Eigenvalues and Eigenvectors ----------------
'''Eigen decomposition: A * v = e * v
*if we multiply A with v the result is a scaler value. A.v=e.v
*eigent vector does not change the direction, just stretched or squashed'''
mat = tf.constant([[1, 2, 3], [2, 3, 5], [6, 6, 3]])
e, v = tf.linalg.eigh(tf.cast(mat, tf.float32))
print("Eigenvalues:", e)
print("Eigenvectors:", v)
print("=============================================")

# ---------------- Band Part ----------------
'''tf.linalg.band_part: extracts band parts of matrix
* input: A tensor of rank ≥ 2 (e.g., 2D matrix or a batch of matrices).
* num_lower: Number of subdiagonals to keep (below the main diagonal).
    If num_lower = -1, keep all lower subdiagonals.
* num_upper: Number of superdiagonals to keep (above the main diagonal).
    If num_upper = -1, keep all upper superdiagonals.'''
mat = tf.constant([[1, 2, 3, 4], [7, 8, 9, 10], [4, 5, 6, 3], [7, 5, 3, 1]])
band_part = tf.linalg.band_part(mat, 1, 2)
print("Band part of matrix:", band_part)
print("=============================================")

# ---------------- Einsum Operations ----------------
'''Einsum allows defining Tensors by defining their element-wise computation. 
This computation is defined by equation, a shorthand form based on Einstein summation. 
As an example, consider multiplying two matrices A and B to form a matrix C. '''
m1 = tf.random.uniform([2, 3], minval=5, maxval=10, dtype=tf.int32)
m2 = tf.random.uniform([3, 4], minval=5, maxval=10, dtype=tf.int32)
m3 = tf.random.uniform([4, 3], minval=5, maxval=10, dtype=tf.int32)
m4 = tf.random.uniform([3, 4], minval=5, maxval=10, dtype=tf.int32)

s1 = tf.random.uniform([3, 3], minval=5, maxval=10, dtype=tf.int32)
s2 = tf.random.uniform([5, 5], minval=5, maxval=10, dtype=tf.int32)

b1 = tf.random.uniform([2, 2, 3], minval=5, maxval=10, dtype=tf.int32)
b2 = tf.random.uniform([2, 3, 4], minval=5, maxval=10, dtype=tf.int32)
b3 = tf.random.uniform([2, 4, 3], minval=5, maxval=10, dtype=tf.int32)

print("Matrix multiplication (Einsum):", tf.einsum("ij,jk->ik", m1, m2))
print("Element-wise multiplication (Einsum):", tf.einsum("ij,ij->ij", s1, s1))
print("Transpose using Einsum:", tf.einsum("ij->ji", m1))
print("Multiple matrix multiplication:", tf.einsum("ab,bc,cd,de->ae", m1, m2, m3, m4))
print("Batch multiplication (3D):", tf.einsum("bij,bjk,bkl->bil", b1, b2, b3))
print("Sum of all elements:", tf.einsum("bij->", b1))
print("Sum of columns:", tf.einsum("bij->bj", b1))
print("Sum of rows:", tf.einsum("bij->bi", b2))

A = tf.random.uniform([2, 4, 2, 4], minval=5, maxval=10, dtype=tf.int32)
B = tf.random.uniform([2, 4, 3, 2], minval=5, maxval=10, dtype=tf.int32)
print("Bucket problem einsum:", tf.einsum("bcij,bcki->bcij", A, B))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf

print("\n========== 1. Tensor Creation ==========")
m1 = tf.random.uniform([3, 3, 3], 2, 100, tf.int32)
m2 = tf.random.uniform([3, 3, 3], 10, 80, tf.int32)
print("Tensor m1:\n", m1.numpy())
print("Tensor m2:\n", m2.numpy())

print("\n========== 2. Expand Dimensions ==========")
print("Expanded m1 at axis=0:\n", tf.expand_dims(m1, axis=0).numpy())

print("\n========== 3. Squeeze Dimensions ==========")
print("Squeezed m1 (removes all dims with size=1):\n", tf.squeeze(m1).numpy())

print("\n========== 4. Reshape ==========")
print("Reshape m1 to shape [3,9]:\n", tf.reshape(m1, [3, 9]).numpy())
print("Reshape m1 to shape [3,-1] (auto cols):\n", tf.reshape(m1, [3, -1]).numpy())

print("\n========== 5. Concatenation ==========")
print("Concatenate m1 & m2 along axis=0 (depth):\n", tf.concat([m1, m2], axis=0).numpy())
print("Concatenate m1 & m2 along axis=1 (rows):\n", tf.concat([m1, m2], axis=1).numpy())
print("Concatenate m1 & m2 along axis=2 (cols):\n", tf.concat([m1, m2], axis=2).numpy())

print("\n========== 6. Stack ==========")
print("Stack m1 & m2 along axis=0:\n", tf.stack([m1, m2], axis=0).numpy())
print("Stack m1 & m2 along axis=1:\n", tf.stack([m1, m2], axis=1).numpy())
print("Stack m1 & m2 along axis=2:\n", tf.stack([m1, m2], axis=2).numpy())

print("\n========== 7. Expand + Concat (like stack) ==========")
print("Expand m1 & m2 at axis=2 and concat:\n", tf.concat([tf.expand_dims(m1, 2), tf.expand_dims(m2, 2)], axis=2).numpy())

print("\n========== 8. Multiple Expand Examples ==========")
print("Expand m1 at axis=0:\n", tf.expand_dims(m1, 0).numpy())
print("Expand m1 at axis=1:\n", tf.expand_dims(m1, 1).numpy())
print("Expand m1 at axis=2:\n", tf.expand_dims(m1, 2).numpy())

print("\n========== 9. Padding ==========")
t1 = tf.random.uniform([3, 3], 1, 10, tf.int32)
print("Original t1:\n", t1.numpy())
print("Pad t1 CONSTANT [[1,2],[3,4]]:\n", tf.pad(t1, [[1, 2], [3, 4]], "CONSTANT").numpy())
print("Pad t1 CONSTANT with value 5:\n", tf.pad(t1, [[1, 2], [3, 4]], "CONSTANT", constant_values=5).numpy())
print("Pad t1 REFLECT:\n", tf.pad(t1, [[1, 1], [1, 1]], "REFLECT").numpy())
print("Pad t1 SYMMETRIC:\n", tf.pad(t1, [[1, 1], [1, 1]], "SYMMETRIC").numpy())

print("\n========== 10. tf.gather (row/column selection) ==========")
t1 = tf.random.uniform([3, 3], 0, 100, tf.int32)
print("Tensor t1:\n", t1.numpy())
print("Select row 3 (index 2):\n", tf.gather(t1, [2], axis=0).numpy())
print("Select row 1 and 3:\n", tf.gather(t1, [0, 2], axis=0).numpy())
print("Select col 3 and 1:\n", tf.gather(t1, [2, 0], axis=1).numpy())

print("\n========== 11. tf.gather_nd (value selection) ==========")
t2 = tf.random.uniform([2, 3, 3], 0, 100, tf.int32)
print("3D Tensor t2:\n", t2.numpy())
print("Select batch 2 (index 1):\n", tf.gather_nd(t2, [1]).numpy())
print("Select batch 2, row 2:\n", tf.gather_nd(t2, [1, 1]).numpy())
print("Select batch 2, row 2, col 2:\n", tf.gather_nd(t2, [1, 1, 1]).numpy())
print("Select elements (1,1,1) and (0,2,2):\n", tf.gather_nd(t2, [[1, 1, 1], [0, 2, 2]]).numpy())

print("Select (1,0) and (1,1) with batch_dims=1:\n", tf.gather_nd(t2, [[1, 0], [1, 1]], batch_dims=1).numpy())
print("Equivalent without batch_dims:\n", tf.gather_nd(t2, [[0, 1, 0], [1, 1, 1]]).numpy())

print("\n========== 12. 4D Tensor tf.gather_nd with batch_dims ==========")
t3 = tf.random.uniform([2, 2, 2, 3], 0, 10, tf.int32)
print("4D Tensor t3:\n", t3.numpy())
print("Gather_nd with batch_dims=1:\n", tf.gather_nd(t3, [[1, 1, 1], [1, 1, 2]], batch_dims=1).numpy())
