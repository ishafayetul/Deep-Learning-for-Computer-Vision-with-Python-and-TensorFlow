import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#-------------------3D Tensor-------------------------------
'''
creating a three dimensional tensor
shape=(3,2,3) means height 3, row 2, column 3 (H,R,C)
'''
threeDtensor=tf.constant([
    [
        [1,2,3],
        [7,8,9]
    ],
    [
        [7,5,3],
        [4,3,2]
    ],
    [
        [1,5,6],
        [7,5,4]
    ]
])

print(threeDtensor)

#-----------------------4D Tensor-------------------------------
'''
Creating a 4D tensor
'''
fourDtensor=tf.constant([
    [
        [
            [1,2,3],
            [3,5,9]
        ],
        [
            [1,5,9],
            [7,3,5]
        ]
    ],
    [
      [
          [1,5,7],
          [7,8,6]
      ],
      [
          [7,5,6],
          [1,2,4]
      ]
      
    ]
])

print(fourDtensor.ndim)

#---------------------------Casting---------------------------------
'''
casting
'''
import numpy as np

arr=np.array([1,2,3])

converted_tensor=tf.convert_to_tensor(arr)
print("converted np to tf",converted_tensor)

#--------------------------Identity Matrix----------------------------------
'''
creating identity matrix
default dtype=float32
'''
eye_tensor=tf.eye(
    num_rows=5,
    num_columns=5,
    batch_shape=[2,] #it will create two 5x5 iMatrix which is a 3D tensor
)
print(eye_tensor)

#--------------------------filled,ones,zeros,size------------------------------------
'''
tf.fill
tf.ones
tf.zeros
tf.size
'''
filled_tensor=tf.fill([2,3],5) #creates a 2x3 tensor filled with 5
print(filled_tensor)

ones_tensor=tf.ones([2,3]) #creates a 2x3 tensor filled with 1
print(ones_tensor)

zero_tensor=tf.zeros([2,3]) #creates a 2x3 tensor filled with 0
print(zero_tensor)


tensor=tf.constant([
    [1,2,3],
    [3,5,6]
])
size=tf.size(tensor) #returns number of elements
print(size)

#------------------------------Random------------------------------------
'''
tf.random.normal
tf.random.unifrom
'''

normal_random_tensor=tf.random.normal( #creates a randomly tensor using mean and stddev
    [2,3],
    mean=100.0,
    stddev=5.0,
    dtype=tf.dtypes.float16
)
print(normal_random_tensor)

unifrom_random_tensor=tf.random.uniform( #creates a randomly tensor between min and max value
    [2,3],
    minval=2,
    maxval=100,
    dtype=tf.dtypes.int32,
    seed=10
)
print(unifrom_random_tensor)
