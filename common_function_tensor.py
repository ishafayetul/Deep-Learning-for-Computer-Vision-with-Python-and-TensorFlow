import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# m1=tf.random.uniform([3,3,3],2,100,tf.int32)
# m2=tf.random.uniform([3,3,3],10,80,tf.int32)
# print(m1,m2)
# print(tf.expand_dims(m1,axis=0)) #adding extra dimension in axis position

# print(tf.squeeze(m1)) #removes all size-1 dimensions
# #print(tf.squeeze(m1,1)) #removes dimension from 1th pos

# print(tf.reshape(m1,[4,6])) #change the shape into 4x6 shape
# print(tf.reshape(m1,[3,-1])) #change the shape into 3xrequired_col shape

# print(tf.concat([m1,m2],0)) #concates by rows - shape must be same
# print(tf.concat([m1,m2],1)) #concates by col - shape must be same
# print(tf.concat([m1,m2],2)) #for 3D tensor axis=0 means depth, 1 means row and 2 means col

# print(tf.stack([m1,m2],0)) #creates a new axis and stack the new tensor, 
#                            #number of created axis depends on number of stacked tensor
# print(tf.stack([m1,m2],1)) 
# print(tf.stack([m1,m2],2))  
# #print(tf.stack([m1,m2],3))  

# print(tf.concat([tf.expand_dims(m1,2),tf.expand_dims(m2,2)],2)) #same as tf.stack()

# print(tf.expand_dims(m1,0))
# print(tf.expand_dims(m1,1))
# print(tf.expand_dims(m1,2))

# t1=tf.random.uniform([3,3],1,10,tf.int32)
# print(t1)
#print(tf.pad(t1,[[1,2],[3,4]],"CONSTANT")) #paddings, [1,2],[3,4] means [before_row,after_row],[before_col,after_col]
#print(tf.pad(t1,[[1,2],[3,4]],"CONSTANT",5))
#print(tf.pad(t1,[[1,1],[1,1]],"REFLECT"))
#print(tf.pad(t1,[[1,1],[1,1]],"SYMMETRIC"))