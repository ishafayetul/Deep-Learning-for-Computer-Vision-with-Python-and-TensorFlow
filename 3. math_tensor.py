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
