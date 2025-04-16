import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 📌 TensorFlow Math Functions - Notes & Examples

import tensorflow as tf

print("==== tf.math.argmax ====")
# 🔹 tf.math.argmax(<tensor>, axis={0/1})
# Returns index of max value along specified axis.
tensor = tf.constant([[10, 20], [30, 5]])
print("Tensor:\n", tensor.numpy())
print("Argmax axis=0 (column-wise):", tf.math.argmax(tensor, axis=0).numpy())  # ➝ [1 0]
print("Argmax axis=1 (row-wise):", tf.math.argmax(tensor, axis=1).numpy())    # ➝ [1 0]

print("\n==== tf.math.argmin ====")
# 🔹 tf.math.argmin(<tensor>, axis={0/1})
# Returns index of min value along specified axis.
print("Argmin axis=0 (column-wise):", tf.math.argmin(tensor, axis=0).numpy())  # ➝ [0 1]
print("Argmin axis=1 (row-wise):", tf.math.argmin(tensor, axis=1).numpy())    # ➝ [0 1]

print("\n==== tf.math.pow ====")
# 🔹 tf.math.pow(tensor1, tensor2)
# Element-wise power: tensor1^tensor2
a = tf.constant([2, 3], dtype=tf.float32)
b = tf.constant([3, 2], dtype=tf.float32)
print("a:", a.numpy())
print("b:", b.numpy())
print("a^b:", tf.math.pow(a, b).numpy())  # ➝ [8.0 9.0]

print("\n==== tf.math.reduce_sum ====")
# 🔹 tf.math.reduce_sum(tensor, axis=None)
# Sum of elements across dimensions
tensor_sum = tf.constant([[1, 2], [3, 4]])
print("Tensor:\n", tensor_sum.numpy())
print("Sum of all elements:", tf.math.reduce_sum(tensor_sum).numpy())         # ➝ 10
print("Sum along axis=0 (columns):", tf.math.reduce_sum(tensor_sum, axis=0).numpy())  # ➝ [4 6]
print("Sum along axis=1 (rows):", tf.math.reduce_sum(tensor_sum, axis=1).numpy())    # ➝ [3 7]

print("\n==== tf.math.sigmoid ====")
# 🔹 tf.math.sigmoid(tensor)
# Sigmoid activation: 1 / (1 + e^(-x))
x = tf.constant([-1.0, 0.0, 1.0])
print("Input:", x.numpy())
print("Sigmoid:", tf.math.sigmoid(x).numpy())  # ➝ [0.2689 0.5 0.7311]
