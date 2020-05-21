import tensorflow as tf
from models.pointer_network.pointer_attention import PointerAttention
from models.pointer_network.pointer_attention import PointerAttentionNoTrainer

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True)

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

class DecoderNoTrainer(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(DecoderNoTrainer, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True)

        # Attention Layers
        self.attention = PointerAttentionNoTrainer(self.dec_units, self.vocab_size)
        
    def call(self, dec_input, dec_hidden, enc_outputs):

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        # Pass through LSTM
        decoder_outputs = self.lstm(dec_input, initial_state = dec_hidden )

        # Compute the pointers
        pointers = self.attention(decoder_outputs, enc_outputs)
        
        return pointers