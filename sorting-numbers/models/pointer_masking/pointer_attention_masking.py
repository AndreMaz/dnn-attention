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

    # Create mask with 0s because at this point there no pointers were created
    # Initially the pointer can point at any position of the input
    mask = tf.zeros_like(dec_input)

    # Iterate over time steps and compute the pointers
    for _, currentStep in enumerate(steps):
      # decoder_prev_hidden shape is [batch_size, features]
      # enc_output shape is [batch_size, timesteps, features]
      # To perform ops between them we need to reshape the decoder_prev_hidden into [batch_size, 1, features]
      decoder_prev_hidden_with_time_dim = tf.expand_dims(currentStep, 1)

      # score shape [batch_size, max_length, 1]
      # we get 1 at the last axis because we are applying score to self.V
      score = self.V(tf.nn.tanh(
          self.W1(decoder_prev_hidden_with_time_dim) + self.W2(enc_outputs)))

      # Remove last dim. Score shape will be [batch_size, max_length]
      score = tf.squeeze(score, axis=2)

      # Apply mask my subtracting a big number from the score
      # Doing this will make make indexes that were previously selected (pointed) extremely improbable to select again
      score -= mask * self.BIG_NUMBER

      # Apply softmax
      attention_pointer = self.attention(score)
      
      # Get the indices to were the pointers point to
      elemToMask = tf.math.argmax(attention_pointer, axis=1)
      # Convert the positions to one-hot encoding
      # One-hot encoding represents the element that was selected (pointed)
      elemToMask = tf.one_hot(elemToMask, self.input_length)

      # Update the mask by adding the one-hot
      mask = mask + elemToMask

      # Store the pointer
      pointerList.append(attention_pointer)
    
    # Convert list back to tensor
    # Will create a time-major Tensor [time_steps, batch_size, features]
    pointerList = tf.convert_to_tensor(pointerList)

    # Put the data back into batch-major shape [batch_size, time_steps, features]
    return tf.transpose(pointerList, [1, 0, 2])

class PointerAttentionNoTrainer(Layer):
  def __init__(self, units, vocab_size):
    super(PointerAttentionNoTrainer, self).__init__()
    self.vocab_size = vocab_size

    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

    self.attention = Softmax(axis=1, name="attention")

  def call(self, dec_output, enc_outputs):
    
    # decoder_prev_hidden shape is [batch_size, features]
    # enc_output shape is [batch_size, timesteps, features]
    # To performs ops between them we need to reshape the decoder_prev_hidden into [batch_size, 1, features]
    decoder_prev_hidden_with_time_dim = tf.expand_dims(dec_output, 1)

    # score shape == [batch_size, max_length, 1]
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is [batch_size, max_length, units]
    score = self.V(tf.nn.tanh(
          self.W1(decoder_prev_hidden_with_time_dim) + self.W2(enc_outputs)))

    # Remove last dim
    score = tf.squeeze(score, axis=2)

    # Apply softmax
    # attention_pointer = self.attention(score)

    # Return logits
    return score