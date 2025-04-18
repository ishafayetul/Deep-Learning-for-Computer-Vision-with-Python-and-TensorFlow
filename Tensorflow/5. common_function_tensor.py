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
