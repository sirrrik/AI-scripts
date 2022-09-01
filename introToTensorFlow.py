import tensorflow as tf
import numpy
# import pandas 
import keras

t = tf.ones([5,5,5,5])

t = tf.reshape(t, [125, -1])

print(t)