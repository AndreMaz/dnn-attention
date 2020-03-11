from tensorflow.keras.layers import Layer, Dense, Softmax
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

class PointerAttentionMasking(Layer):
  def __init__(self, units, input_length):
    super(PointerAttentionMasking, self).__init__()
    self.input_length = input_length

    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

    self.attention = Softmax(axis=1, name="attention")

    self.BIG_NUMBER = 1e6
  
  def call(self, dec_outputs, enc_outputs, dec_input):
    # Unstack the input along time axis, i.e, axis = 1
    # Original data is a Tensor of  [ batch_size, time_steps, features] shape
    # After unstacking it is going to be a list (length = time_steps) of Tensors
    # Each element in list is going to be a Tensor of [ batch_size, features] shape
    steps = tf.unstack(dec_outputs, axis=1)
    pointerList = []

    
    # maskSize = len(steps)
    
    mask = tf.zeros_like(dec_input)

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

      # Apply mask
      score -= mask * self.BIG_NUMBER

      # Apply softmax
      attention_pointer = self.attention(score)
      
      elemToMask = tf.math.argmax(attention_pointer, axis=1)
      elemToMask = tf.one_hot(elemToMask, self.input_length)

      mask = mask + elemToMask
      

      # Store the pointer
      pointerList.append(attention_pointer)
    
    # Convert list back to tensor
    # Will create a time-major Tensor [time_steps, batch_size, features]
    pointerList = tf.convert_to_tensor(pointerList)

    # Put the data back into batch-major shape [batch_size, time_steps, features]
    return tf.transpose(pointerList, [1, 0, 2])
