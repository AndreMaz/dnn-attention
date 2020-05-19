from tensorflow.keras.layers import Layer, Dense, Softmax
import tensorflow as tf
import numpy as np

class PointerAttention(Layer):
  def __init__(self, units, vocab_size):
    super(PointerAttention, self).__init__()
    self.vocab_size = vocab_size

    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

    self.attention = Softmax(axis=1, name="attention")

  def call(self, dec_outputs, enc_outputs):
    
    # Unstack the input along time axis, i.e, axis = 1
    # Original data is a Tensor of  [ batch_size, time_steps, features] shape
    # After unstacking it is going to be a list (length = time_steps) of Tensors
    # Each element in list is going to be a Tensor of [ batch_size, features] shape
    steps = tf.unstack(dec_outputs, axis=1)
    pointerList = []

    # Iterate over time steps and compute the pointers
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
      # pointer = tf.nn.softmax(score, axis=1)
      attention_pointer = self.attention(score)
      
      # Store the pointer
      pointerList.append(attention_pointer)
    
    # Convert list back to tensor
    # Will create a time-major Tensor [time_steps, batch_size, features]
    pointerList = tf.convert_to_tensor(pointerList)

    # Put the data back into batch-major shape [batch_size, time_steps, features]
    return tf.transpose(pointerList, [1, 0, 2])
