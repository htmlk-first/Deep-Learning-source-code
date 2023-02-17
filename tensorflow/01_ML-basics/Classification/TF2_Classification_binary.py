
import tensorflow as tf
import numpy as np

x_data = np.array([[1,2],[2,3],[3,4],[4,3],[5,3],[6,2]], dtype=np.float32)
y_data = np.array([[0],[0],[0],[1],[1],[1]], dtype=np.float32)

W = tf.Variable(tf.random.normal([2,1]))
b = tf.Variable(tf.random.normal([1]))

def run_optimization():
  with tf.GradientTape() as g:
    model = tf.sigmoid(tf.matmul(x_data,W)+b)
    cost = tf.reduce_mean((-1)*y_data*tf.math.log(model)+(-1)*(1-y_data)*tf.math.log(1-model))
  gradients = g.gradient(cost,[W,b])
  tf.optimizers.SGD(0.01).apply_gradients(zip(gradients,[W,b]))

for step in range(2001):
  run_optimization()

model = tf.sigmoid(tf.matmul(x_data,W)+b)
cost = tf.reduce_mean((-1)*y_data*tf.math.log(model)+(-1)*(1-y_data)*tf.math.log(1-model))
prediction = tf.cast(model>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_data),dtype=tf.float32))
print("Model", model.numpy())
print("Prediction", prediction.numpy())
print("Accuracy: ", accuracy.numpy())
