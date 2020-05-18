from tensorflow.keras.layers import Embedding, LSTM, Layer
# from tensorflow.keras import Model

class Encoder(Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.enc_units,
                       return_sequences=True,
                       return_state=True)

    def call(self, x):
        # Get the embeddings for the input
        x = self.embedding(x)

        # Run in trough the LSTM
        out, hidden_state, carry_state = self.lstm(x)
        return out, hidden_state, carry_state
