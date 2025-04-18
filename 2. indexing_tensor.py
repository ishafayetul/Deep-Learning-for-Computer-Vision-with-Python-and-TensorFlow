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
