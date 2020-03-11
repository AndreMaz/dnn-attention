from tensorflow.keras.layers import Layer, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
import tensorflow as tf

class LuongAttention(Layer):
  def __init__(self):
    super(LuongAttention, self).__init__()
    
    self.attentionDot = Dot((2, 2), name="attentionDot")

    self.attention_layer = Activation("softmax", name="attentionSoftMax")

    self.context = Dot((2, 1), name="context")


  def call(self, decoderLSTMOutput, encoderLSTMOutput):

    attention = self.attentionDot([decoderLSTMOutput, encoderLSTMOutput])

    attention = self.attention_layer(attention)

    context_vector = self.context([attention, encoderLSTMOutput])

    return context_vector, attention