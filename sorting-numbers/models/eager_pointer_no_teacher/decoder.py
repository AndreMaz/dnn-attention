import tensorflow as tf
from models.eager_pointer.pointer_attention import PointerAttention
from tensorflow.keras.layers import Embedding, LSTM, Layer


class Decoder(Layer):
    def __init__(self, SOS_CODE, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.SOS_CODE = SOS_CODE
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = Embedding(vocab_size, embedding_dim)

        self.lstm = LSTM(self.dec_units, return_sequences=True)

        # Attention Layers
        self.attention = PointerAttention(self.dec_units, self.vocab_size)
        
    def call(self, dec_hidden, enc_outputs):
        
        # Decoder's input starts with SOS code
        sos_tensor = tf.fill([enc_outputs.shape[0], 1], self.SOS_CODE)
        # Remaining are set to zero
        zeroed_tensor = tf.zeros([enc_outputs.shape[0], enc_outputs.shape[1] - 1], dtype="int32")
        
        # Create the actual decoder's input
        # shape = [batch_size, sequence_size]
        dec_input = tf.concat([sos_tensor, zeroed_tensor], 1)

        # Convert input to embeddings
        dec_input = self.embedding(dec_input)

        # Pass through LSTM
        decoder_outputs = self.lstm(dec_input, initial_state = dec_hidden)

        # Compute the pointers
        pointers = self.attention(decoder_outputs, enc_outputs)
        
        return pointers