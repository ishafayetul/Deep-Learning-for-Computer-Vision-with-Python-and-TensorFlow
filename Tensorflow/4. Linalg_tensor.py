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
