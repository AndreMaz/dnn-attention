from tensorflow.keras.layers import Layer, Dense
import tensorflow as tf

class BahdanauAttention(Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, decoder_prev_hidden, enc_outputs):
    # decoder_prev_hidden shape is [batch_size, features]
    # enc_output shape is [batch_size, timesteps, features]
    # To performs ops between them we need to reshape the decoder_prev_hidden into [batch_size, 1, features]
    decoder_prev_hidden_with_time_dim = tf.expand_dims(decoder_prev_hidden, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(decoder_prev_hidden_with_time_dim) + self.W2(enc_outputs)))

    # Apply softmax
    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * enc_outputs
    # Sum along 1 axis to get [batch_size, hidden_size] shape
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights