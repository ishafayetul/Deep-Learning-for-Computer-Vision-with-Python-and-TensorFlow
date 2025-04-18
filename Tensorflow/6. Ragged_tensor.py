import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
import numpy as np

#----------------- 01. Creating Ragged Tensor -----------------------
'''Not regtangular tensor having none col'''
rt1=tf.ragged.constant([
    [1,2,3,4],
    [3,5],
    [4],
    [3,2,1]
])
#print(rt1,rt1.shape)

#---------------- 02. Slicing --------------
'''Slicing can be done like regular tensor'''
# print(rt1[0,3])#1st row 4th element
# print(rt1[1,1])#2nd row 2nd element
# print(rt1[0:2,0:2])#Upto 2nd row, Upto 2nd Element
# print(rt1[3,1:2])#3rd row 2nd element

#-------- 03. Concat----------------

rt2=tf.ragged.constant([
    [4,5,6],
    [3,5]
])
#print(tf.concat([rt1,rt2],axis=0)) #rt2 will concat under rt1, shape does not matter : [[1, 2, 3, 4], 
                                                                                       # [3, 5], [4],
                                                                                       # [3, 2, 1], 
                                                                                       # [4, 5, 6], 
                                                                                       # [3, 5]]

rt3=tf.ragged.constant([
    [-10,-11,-16],
    [-13,-15]
])
#print(tf.concat([rt2,rt3],axis=1)) #rt3 will concat beside rt2, shape matters : [[4, 5, 6, -4, -5, -6], 
                                                                               # [3, 5, -3, -5]]
#------------ 04. Math (Element wise) -----------
# print(rt2+rt3) #Sum
# print(rt2-rt3) #Sub                                                                             
# print(rt2*rt3) #Mul
# print(rt2/rt3) #div

#------------------- 05. Convertion & Masking ------------
'''Converted to regular tensor filled with given value known as Masking'''
t1=rt1.to_tensor(default_value=-1)
#print(t1)

#---------------- 06. Nested Tensor -------------
'''ragged tensor into another ragged tensor'''
# nrt=tf.ragged.constant([
#     [[1, 2], [3, 4]],
#     [[5, 6]],
#     [[7, 8], [9, 10]]
# ])
#print(nrt)

#---------- 07. Flatenning ------------------
'''converted to a 1D tensor'''
# ft=nrt.flat_values
# print(ft)

#------------- 08. Flat Mapping --------------
# mapped=tf.ragged.map_flat_values(lambda x: x*3,rt1)
# print(mapped)

#--------------- 08. Boolean Masking ------------
'''selecting elements based on a corresponding boolean tensor 
Both data and mask should be the same shape or broadcastable'''
mask=tf.ragged.constant([
    [False,True,False,True],
    [True,False],
    [False],
    [True,True,False]
])

print(rt1)
print(tf.ragged.boolean_mask(rt1,mask)) #shape of tensor and mask are same
print(tf.ragged.boolean_mask(rt1,[True,False,True,False])) #masking whole rows

#tf.RaggedTensor
#tf.Variable