from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras import Model
from tensorflow import zeros


class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.enc_units,
                       return_sequences=True,
                       return_state=True)

    def call(self, x):
        x = self.embedding(x)

        # out, hidden_state, carry_state = self.lstm(x, initial_state = [hidden, hidden])
        out, hidden_state, carry_state = self.lstm(x)
        return out, hidden_state, carry_state

    # def initialize_hidden_state(self):
    #     return zeros((self.batch_sz, self.enc_units))