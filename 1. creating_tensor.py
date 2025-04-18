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
