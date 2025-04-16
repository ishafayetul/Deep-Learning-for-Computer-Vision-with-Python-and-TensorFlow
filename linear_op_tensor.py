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

t3=tf.constant([
    [1,3,5],
    [2,4,6],
    [3,6,9]
])
#----------------Adjoint-------------------------
adjoint=tf.linalg.adjoint(t2)
print(adjoint)
print(tf.linalg.inv(tf.cast(t2,tf.float32)))
print(tf.linalg.det(tf.cast(t2,tf.float32)))
print("==========================================")
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

cross_t2xt3=tf.linalg.cross(t2,t3) #element wise cross product
print(cross_t2xt3)
print("======================================================")
#Matrix Decomposition-------------------------------------------------------------------------

#--------------------Choleskey Decomposition-----------------------
'''
cholesky decomposition, matrix must be sq and symmetric, it decomposes
the matrix in a way that [A]=[L].[L]^T , where L is a lower triangular matrix
'''
sym_tensor=tf.constant([ 
    [4,2],
    [2,3]
])

cholesky_decomp=tf.linalg.cholesky(tf.cast(sym_tensor,tf.float32))
print(cholesky_decomp)

print(cholesky_decomp@tf.transpose(cholesky_decomp))
print("======================================================")
#-----------------------QR Decomposition------------------------------
'''
[A]=[Q].[R] || Q= orthogonal mat, R= Upper Triangular mat
*Orthogonal Mat: [A] is an orthogonal mat if [A]^T=[A]^-1 
* return two tensors Q & R
* [A] must not be int
'''
tensor=tf.constant([
    [1.0,2.0,3.0],
    [3.0,5.0,6.0],
    [4.0,7.0,-9.0]
])
q,r=tf.linalg.qr(tensor,True)
print(q)
print(r)
print(q@r)
#q@r=tensor
print("======================================================")
#------------------------------Singular Value Decomposition (SVD)-----------------------------
'''
Decomposes a MxN matrix such that: [A]=[S].[U].[V]^t
*S= MxN Diagonal Matrix ->  its diagonal values are singular values of A
*U= MxM orthogonal matrix -> its columns are left singular vectors of A
*V= NxN orthogonal Matrix ->  its columns are right singular vectors of A
'''

mat=tf.constant([
    [1.,2.,3.],
    [4.,5.,6.],
    [7.,8.,9.]
])
s,u,v=tf.linalg.svd(mat,full_matrices=True)
print(s)
print(u)
print(v)
print("===============================================")
#--------------------Eigenvalues & EigenVector-------------------------
'''
Computes eigenvalue and eigenvector of a sq mat, returns e,v
*if we multiply A with v the result is a scaler value. A.v=e.v
*eigent vector does not change the direction, just stretched or squashed
'''
mat=tf.constant([
    [1,2,3],
    [2,3,5],
    [6,6,3]
])

e,v=tf.linalg.eigh(tf.cast(mat,tf.float32))
print(e)
print(v)
print("=============================================")
#-------------------------------------Band Part-----------------------
'''
--> tf.linalg.band_part(input, num_lower, num_upper)
* input: A tensor of rank ≥ 2 (e.g., 2D matrix or a batch of matrices).
* num_lower: Number of subdiagonals to keep (below the main diagonal).
    If num_lower = -1, keep all lower subdiagonals.
* num_upper: Number of superdiagonals to keep (above the main diagonal).
    If num_upper = -1, keep all upper superdiagonals.
'''
mat=tf.constant([
    [1,2,3,4],
    [7,8,9,10],
    [4,5,6,3],
    [7,5,3,1]
])
band_part=tf.linalg.band_part(mat,1,2)
print(band_part)
print("=============================================")
#---------------------------------Eisum-------------------------------
'''
Einsum allows defining Tensors by defining their element-wise computation. 
This computation is defined by equation, a shorthand form based on Einstein summation. 
As an example, consider multiplying two matrices A and B to form a matrix C. 
'''
m1=tf.random.uniform([2,3],minval=5,maxval=10,dtype=tf.int32)
m2=tf.random.uniform([3,4],minval=5,maxval=10,dtype=tf.int32)
m3=tf.random.uniform([4,3],minval=5,maxval=10,dtype=tf.int32)
m4=tf.random.uniform([3,4],minval=5,maxval=10,dtype=tf.int32)

s1=tf.random.uniform([3,3],minval=5,maxval=10,dtype=tf.int32)
s2=tf.random.uniform([5,5],minval=5,maxval=10,dtype=tf.int32)

b1=tf.random.uniform([2,2,3],minval=5,maxval=10,dtype=tf.int32)
b2=tf.random.uniform([2,3,4],minval=5,maxval=10,dtype=tf.int32)
b3=tf.random.uniform([2,4,3],minval=5,maxval=10,dtype=tf.int32)

mul=tf.einsum("ij,jk->ik",m1,m2) #mat mul
print(mul)

hc=tf.einsum("ij,ij->ij",s1,s1) #element wise multiplication
print(hc)

transpose=tf.einsum("ij->ji",m1) #transpose 
print(transpose)

multiple_mul=tf.einsum("ab,bc,cd,de -> ae",m1,m2,m3,m4) #multiple mat mul
print(multiple_mul)

batch_mul=tf.einsum("bij,bjk,bkl -> bil",b1,b2,b3) #3D mul/batch mul where b is height
print(batch_mul)

sum=tf.einsum("bij->",b1) #sum of all element of a mat
print(sum)

sumC=tf.einsum("bij->bj",b1) #sum of all elements of col
print(sumC)

sumR=tf.einsum("bij->bi",b2) #sum off all elements of row
print(sumR)

A=tf.random.uniform([2,4,2,4],minval=5,maxval=10,dtype=tf.int32)
B=tf.random.uniform([2,4,3,2],minval=5,maxval=10,dtype=tf.int32)

ab=tf.einsum("bcij,bcki -> bcij",A,B) #bucket problem
print(ab)