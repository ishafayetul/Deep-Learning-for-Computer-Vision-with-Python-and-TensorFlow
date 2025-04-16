import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

t1=tf.constant([
    [1,2,3],
    [4,7,6]
])
t2=tf.constant([
    [7,8,5],
    [8,9,3],
    [4,5,1]
])

#Matrix Operations-----------------------------------------------------------------------------

t1xt2=tf.linalg.matmul(t1,t2) #Matrix multiplication
print(t1xt2)
print(t1@t2)

t2_trace=tf.linalg.trace(t1) #trace (sum of main diagonal's element)
print(t2_trace)

t2_det=tf.linalg.det(tf.cast(t2,tf.dtypes.float32)) #determinent of a sq mat, tensor type must be float
print(t2_det)

t2_inv=tf.linalg.inv(tf.cast(t2,tf.float32)) #inv mat of a sq mat, tensor type must be float
print(t2_inv)

t1_transpose=tf.linalg.matrix_transpose(t1) #transpose of a matrix
print(t1_transpose)

#Matrix Decomposition-------------------------------------------------------------------------
sym_tensor=tf.constant([ #cholesky decomposition, matrix must be sq and symmetric
    [4,2],
    [2,3]
])

cholesky_decomp=tf.linalg.cholesky(tf.cast(sym_tensor,tf.float32))
print(cholesky_decomp)

print(cholesky_decomp@tf.transpose(cholesky_decomp))