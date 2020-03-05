import tensorflow as tf
from models.pointer_network.pointer_attention import PointerAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Attention Layers
        self.attention = PointerAttention(self.dec_units, self.vocab_size)

        # We are going to do the looping manually so instead of LSMT Layer we use LSTM cell
        # self.cell = tf.keras.layers.LSTMCell(
        #     self.dec_units, recurrent_initializer='glorot_uniform')

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                       return_sequences=True)
        
        self.V = tf.keras.layers.Dense(11)

    def call(self, dec_input, dec_hidden, enc_outputs):

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        decoder_outputs = self.lstm(dec_input, initial_state = dec_hidden )

        # attention = tf.keras.layers.Dot((2, 2), name="attentionDot")(
        #     [decoder_outputs, enc_outputs])

        # a = self.V(attention)
        # return a

        pointers = self.attention(decoder_outputs, enc_outputs)
        return pointers

        # return 1
