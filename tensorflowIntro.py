from numpy import dtype
import tensorflow as tf
import keras
import numpy as np


model = keras.Sequential([keras.layers.Dense(units = 1, input_shape=[1])])

# this sets the activation function as sigmiod and the loss function 
model.compile(optimizer='sgd', loss='mean_squared_error')

#  the dataset 
xs = np.array([-1.0,0.0,1.0, 2.0,3.0,4.0], dtype=float )

ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# this line trains the model on the above data 500 times
model.fit(xs, ys, epochs=500)

# given a y =10 predict x ans = 18.9
print(model.predict([10.0]))

