import tensorflow as tf
import numpy as np
from models.transformer_masking.decoder_layer import DecoderLayer
from models.transformer_masking.utils import positional_encoding

from models.transformer_masking.last_decoder_layer import LastDecoderLayer

class Decoder(tf.keras.layers.Layer):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               SOS_CODE,
               target_vocab_size,
               maximum_position_encoding,
               rate=0.1):
    super(Decoder, self).__init__()

    self.SOS_CODE = SOS_CODE

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

    self.last_decoder_layer = LastDecoderLayer(d_model, num_heads, dff, rate)

    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self,
           enc_input,
           enc_output,
           training,
           look_ahead_mask,
           padding_mask):

    batch_size = enc_input.shape[0]
    # Create a tensor with the batch indices
    batch_indices = tf.convert_to_tensor(
            list(range(batch_size)), dtype='int32')

    # Decoder's input starts with SOS code
    dec_input = tf.fill([batch_size, 1], self.SOS_CODE)
    # In the beggining this is 
    # dec_input = sos_tensor 

    # Number of pointers to generate
    num_steps = tf.shape(enc_input)[1]

    for step in range(num_steps):

      x = dec_input ### Decoder's input

      seq_len = tf.shape(x)[1]
      attention_weights = {}
      
      x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      x += self.pos_encoding[:, :seq_len, :]
      
      x = self.dropout(x, training=training)

      for i in range(self.num_layers):
        x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                              look_ahead_mask, padding_mask)
        
      #  attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      #  attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
      
      combined_attention = self.last_decoder_layer(x, enc_output, training, look_ahead_mask, padding_mask)

      pointer_index = combined_attention.numpy().argmax(2)[:, -1]
      pointed_value = enc_input.numpy()[batch_indices, pointer_index]
      pointed_value = np.reshape(pointed_value, (batch_size, 1))
      # Update decoder input
      dec_input = tf.concat([dec_input, pointed_value], axis=-1)

    # x.shape == (batch_size, target_seq_len, d_model)
    return combined_attention
