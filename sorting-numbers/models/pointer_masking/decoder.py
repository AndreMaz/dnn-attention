import tensorflow as tf
from models.pointer_masking.pointer_attention_masking import PointerAttentionMasking


class Decoder(tf.keras.Model):
    def __init__(self, input_length, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.input_length = input_length

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True)

        # Attention Layers
        self.attention = PointerAttentionMasking(self.dec_units, self.input_length)
        
    def call(self, dec_input, dec_hidden, enc_outputs):

        # Convert input to embeddings
        dec_embedding_output = self.embedding(dec_input)

        # Pass through LSTM
        decoder_outputs = self.lstm(dec_embedding_output, initial_state = dec_hidden )

        # Compute the pointers
        pointers = self.attention(decoder_outputs, enc_outputs, dec_input)
        
        return pointers