import tensorflow as tf
from models.transformer_masking.decoder import Decoder
from models.transformer_masking.encoder import Encoder

class Transformer(tf.keras.Model):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               input_vocab_size,
               target_vocab_size,
               SOS_CODE,
               pe_input,
               pe_target,
               rate=0.1
               ):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           input_vocab_size,
                           pe_input,
                           rate)

    self.decoder = Decoder(num_layers, 
                           d_model,
                           num_heads,
                           dff,
                           SOS_CODE,
                           target_vocab_size,
                           pe_target,
                           rate)

    # self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self,
           encoder_input,
           decoder_input,
           training: bool,
           enc_padding_mask,
           look_ahead_mask,
           dec_padding_mask
           ):
    
    # enc_output.shape = (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(encoder_input,
                              training,
                              enc_padding_mask)
    
    # Returns attentions with pointer locations
    # combined_attention.shape == (batch_size, inp_seq_len, inp_seq_len)
    combined_attention = self.decoder(decoder_input,
                                      encoder_input,
                                      enc_output,
                                      training,
                                      look_ahead_mask,
                                      dec_padding_mask)

    return combined_attention