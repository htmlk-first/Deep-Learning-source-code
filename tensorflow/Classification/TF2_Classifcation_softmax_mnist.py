import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.datasets import mnist

batch_size = 128
nH1 = 256
nH2 = 256
nH3 = 256

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32').reshape([-1, 784]) / 255.
x_test = x_test.astype('float32').reshape([-1, 784]) / 255.
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)

class ANN(object):
  def __init__(self):
    self.W_1 = tf.Variable(tf.random.normal(shape=[784, nH1]))
    self.W_2 = tf.Variable(tf.random.normal(shape=[nH1, nH2]))
    self.W_3 = tf.Variable(tf.random.normal(shape=[nH2, nH3]))
    self.W_Out = tf.Variable(tf.random.normal(shape=[nH3, 10]))
    self.b_1 = tf.Variable(tf.random.normal(shape=[nH1]))
    self.b_2 = tf.Variable(tf.random.normal(shape=[nH2]))
    self.b_3 = tf.Variable(tf.random.normal(shape=[nH3]))
    self.b_Out = tf.Variable(tf.random.normal(shape=[10]))

  def __call__(self, x):
    H1_Out = tf.nn.relu(tf.matmul(x, self.W_1) + self.b_1)
    H2_Out = tf.nn.relu(tf.matmul(H1_Out, self.W_2) + self.b_2)
    H3_Out = tf.nn.relu(tf.matmul(H2_Out, self.W_3) + self.b_3)
    Out = tf.matmul(H2_Out, self.W_Out) + self.b_Out
    return Out

ANN_model = ANN()
optimizer = tf.optimizers.Adam(0.01)

@tf.function
def run_optimization(model, x, y):
  with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= y_pred, labels=y))
  gradients = tape.gradient(loss, vars(model).values())
  optimizer.apply_gradients(zip(gradients, vars(model).values()))

for epoch in range(100):
  avg_loss = 0
  tot_batch = int(x_train.shape[0] / batch_size)
  for batch_x, batch_y in train_data:
    run_optimization(ANN_model, batch_x, batch_y)
    current_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ANN_model(batch_x), labels=batch_y))
    avg_loss += current_loss / tot_batch
  if epoch % 1 == 0:
    print("Step: %d, Loss: %f" % ((epoch), avg_loss))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ANN_model(x_test),1), tf.argmax(y_test,1)), tf.float32))
print("Accuracy: %f", accuracy) 