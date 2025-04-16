import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tensor=tf.constant([
    [1,2,3],
    [5,6,7],
    [8,9,10]
])

print(tensor[:,:]) #prints all
print(tensor[0,:]) #1st row
print(tensor[:,0]) #1st col
print(tensor[:1,:1]) #1st element
print(tensor[:2,:3]) #1st two row
print(tensor[1:2,:3]) #2nd row
print(tensor[:3,1:2]) #2nd col
print(tensor[2:3,2:3])#last element
print(tensor[0:-2,0:-2])

    