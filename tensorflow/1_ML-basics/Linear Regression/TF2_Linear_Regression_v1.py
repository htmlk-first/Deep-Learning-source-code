import tensorflow as tf
import numpy as np

x_data = np.array([[1,1],[2,2],[3,3]], dtype=np.float32)
y_data = np.array([[10],[20],[30]], dtype=np.float32)

W = tf.Variable(tf.random.normal([2,1]))
b = tf.Variable(tf.random.normal([1]))

def run_optimization():
  with tf.GradientTape() as g:
    model = tf.matmul(x_data,W)+b 
    cost = tf.reduce_mean(tf.square(model-y_data))
  gradients = g.gradient(cost,[W,b])
  tf.optimizers.SGD(0.01).apply_gradients(zip(gradients,[W,b]))

for step in range(2001):
  run_optimization()

print(W.numpy())
print(b.numpy())
x_test = np.array([[4,4]],dtype=np.float32)
model_test = tf.matmul(x_test, W)+b
print("model for [4,4]: ", model_test.numpy())
