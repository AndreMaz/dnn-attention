from tensorflow.keras.layers import Layer, Dense, TimeDistributed
import tensorflow as tf
import numpy as np

class PointerAttention(Layer):
  def __init__(self, units):
    super(PointerAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, dec_outputs, enc_outputs):
    steps = tf.unstack(dec_outputs, axis=1)
    pointerList = []

    for _, currentStep in enumerate(steps):
      # decoder_prev_hidden shape is [batch_size, features]
      # enc_output shape is [batch_size, timesteps, features]
      # To performs ops between them we need to reshape the decoder_prev_hidden into [batch_size, 1, features]
      decoder_prev_hidden_with_time_dim = tf.expand_dims(currentStep, 1)

      # score shape == (batch_size, max_length, 1)
      # we get 1 at the last axis because we are applying score to self.V
      # the shape of the tensor before applying self.V is (batch_size, max_length, units)
      score = self.V(tf.nn.tanh(
          self.W1(decoder_prev_hidden_with_time_dim) + self.W2(enc_outputs)))

      # Remove last dim
      score = tf.squeeze(score, axis=2)

      # Apply softmax
      attention_weights = tf.nn.softmax(score, axis=1)
      # Find the pointer
      pointer = tf.argmax(attention_weights, axis = 1)

      # Encoder in into one-hot
      pointer = tf.one_hot(pointer, 10)
      pointerList.append(pointer)
    
    # pointer[index] = 1

    pointerList = tf.convert_to_tensor(pointerList)
    # context_vector = attention_weights * enc_outputs
    # Sum along 1 axis to get [batch_size, hidden_size] shape
    # context_vector = tf.reduce_sum(context_vector, axis=1)

    return tf.transpose(pointerList, [1, 0, 2])
