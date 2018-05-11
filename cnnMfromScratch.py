
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):

  print("model called")
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 200, 200, 16])
  input_Shape= input_layer.get_shape().as_list()
  # print(input_Shape)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  is_train=True 
  if mode == tf.estimator.ModeKeys.TRAIN:
    is_train= True
  else:
    is_train=False
  batchnorm0=  tf.contrib.layers.batch_norm(input_layer, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn0')


  dropout0 = tf.layers.dropout(
      inputs=batchnorm0, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)


  conv1 = tf.layers.conv2d(
      inputs=dropout0,
      filters=96,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)
  conv1_Shape=conv1.get_shape().as_list()
  # print(conv1_Shape)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 224, 224, 32]
  # Output Tensor Shape: [batch_size, 112, 112, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  pool1_shape=pool1.get_shape().as_list()
  # print(pool1_shape)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 112, 112, 32]
  # Output Tensor Shape: [batch_size, 112, 112, 64]
  batchnorm2=  tf.contrib.layers.batch_norm(pool1, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn2')

  conv2 = tf.layers.conv2d(
      inputs=batchnorm2,
      filters=256,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  conv2_shape=conv2.get_shape().as_list()
  # print(conv2_shape)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 112, 112, 64]
  # Output Tensor Shape: [batch_size, 56, 56, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  batchnorm3=  tf.contrib.layers.batch_norm(pool2, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn3')


  conv3=tf.layers.conv2d(
    inputs= batchnorm3,
    filters=512,
    kernel_size=[3,3],
    activation=tf.nn.relu,
    padding="same"
    )
  batchnorm4=  tf.contrib.layers.batch_norm(conv3, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn4')


  conv4=tf.layers.conv2d(
      inputs= batchnorm4,
      filters=512,
      kernel_size=[3,3],
      activation=tf.nn.relu,
      padding="same"
      )
  batchnorm5=  tf.contrib.layers.batch_norm(conv4, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn5')


  conv5=tf.layers.conv2d(
      inputs= batchnorm5,
      filters=512,
      kernel_size=[3,3],
      activation=tf.nn.relu,
      padding="same"
      )
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)



  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 56, 56, 64]
  # Output Tensor Shape: [batch_size, 56 * 56 * 64]
  pool5Shape = pool5.get_shape().as_list()
  # print(pool5Shape)
  pool5_flat = tf.reshape(pool5, [-1, pool5Shape[1]*pool5Shape[2]*pool5Shape[3]])

  pool5_flatShape = pool5_flat.get_shape().as_list()
  # print(pool5_flatShape)
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  batchnorm6=  tf.contrib.layers.batch_norm(pool5_flat, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn6')


  dense1 = tf.layers.dense(inputs=batchnorm6, units=4096, activation=tf.nn.relu)
  # print(dense1.get_shape().as_list())

  # Add dropout operation; 0.6 probability that element will be kept
  batchnorm7=  tf.contrib.layers.batch_norm(dense1, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn7')

  dropout = tf.layers.dropout(
      inputs=batchnorm7, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs=dropout, units=4096, activation=tf.nn.relu)
  # print(dense2.get_shape().as_list())


  batchnorm8=  tf.contrib.layers.batch_norm(dense2, 
                                          center=True, scale=True, 
                                          is_training=is_train,
                                          scope='bn8')

  # Add dropout operation; 0.6 probability that element will be kept
  dropout2 = tf.layers.dropout(
      inputs=batchnorm8, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)




  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout2, units=3)
  # print(logits.get_shape().as_list())
  # print(labels.get_shape().as_list())

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  # print(logits)
  # print(labels)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.9, beta2=0.99)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
