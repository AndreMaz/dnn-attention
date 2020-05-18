# import tensorflow as tf
from models.eager_pointer.pointer_attention import PointerAttention
from tensorflow.keras.layers import Embedding, LSTM, Layer


class Decoder(Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = Embedding(vocab_size, embedding_dim)

        self.lstm = LSTM(self.dec_units, return_sequences=True)

        # Attention Layers
        self.attention = PointerAttention(self.dec_units, self.vocab_size)
        
    def call(self, dec_input, dec_hidden, enc_outputs):

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        # Pass through LSTM
        decoder_outputs = self.lstm(dec_input, initial_state = dec_hidden )

        # Compute the pointers
        pointers = self.attention(decoder_outputs, enc_outputs)
        
        return pointers