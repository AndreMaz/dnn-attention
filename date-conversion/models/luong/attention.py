from tensorflow.keras.layers import Layer, Dense, Softmax, Dot, Activation
import tensorflow as tf

class LuongAttention(Layer):
  def __init__(self):
    super(LuongAttention, self).__init__()
    
    self.attentionDot = Dot((2, 2), name="attentionDot")

    # self.attention_layer = Activation("softmax", name="attention")

    self.context = Dot((2, 1), name="context")


  def call(self, decoder_output, encoder_output):
    score = self.attentionDot([decoder_output, encoder_output])

    attention_weights = tf.nn.softmax(score, axis=1) #self.attention_layer(attention)

    context_vector = self.context([attention_weights, encoder_output])


    return context_vector, attention_weights