import tensorflow as tf
from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import point_wise_feed_forward_network
from models.transformer.custom_attention import PointerMultiHeadAttention
from models.transformer.custom_attention import PointerAttention

class LastDecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(LastDecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = PointerMultiHeadAttention(d_model, num_heads)

    self.pointer_attention = PointerAttention()

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    combined_attention = self.pointer_attention(out1, enc_output)

    return combined_attention