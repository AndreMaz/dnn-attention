import tensorflow as tf
from models.luong.attention import LuongAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True,
                       return_state=False)
        
        self.attention = LuongAttention()

        self.concat = tf.keras.layers.Concatenate(name="combinedContext")

    def call(self, dec_input, dec_init_state, encoder_output):
         # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        # Run in trough the LSTM
        decoder_output = self.lstm(dec_input, initial_state=[dec_init_state, dec_init_state])
        
        
        context_vector, attention = self.attention(decoder_output, encoder_output)
        
        decoderCombinedContext = self.concat(([context_vector, decoder_output]))

        return decoderCombinedContext