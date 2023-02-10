import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)

W_h = tf.Variable(tf.random.normal([2,3]))
b_h =tf.Variable(tf.random.normal([3]))

W_o = tf.Variable(tf.random.normal([3,1]))
b_o =tf.Variable(tf.random.normal([1]))

def run_optimization():
  with tf.GradientTape() as g:
    H1 = tf.sigmoid(tf.matmul(x_data,W_h)+b_h)
    model = tf.sigmoid(tf.matmul(H1,W_o)+b_o)
    cost = tf.reduce_mean((-1)*y_data*tf.math.log(model)+(-1)*(1-y_data)*tf.math.log(1-model))
  gradients = g.gradient(cost,[W_o,W_h,b_o,b_h])
  tf.optimizers.SGD(0.1).apply_gradients(zip(gradients,[W_o,W_h,b_o,b_h]))

for step in range(20001):
  run_optimization()

H1 = tf.sigmoid(tf.matmul(x_data,W_h)+b_h)
model = tf.sigmoid(tf.matmul(H1,W_o)+b_o)
prediction = tf.cast(model>0.5,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_data),dtype=tf.float32))
print("Model", model.numpy())
print("Prediction", prediction.numpy())
print("Accuracy: ", accuracy.numpy())