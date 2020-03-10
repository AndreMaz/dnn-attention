from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras import Model
from tensorflow import zeros
from models.luong.last_time_step_layer import GetLastTimestepLayer


class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, inputLength):
        super(Encoder, self).__init__()
        self.enc_units = enc_units

        self.embedding = Embedding(
            vocab_size,
            embedding_dim,
            input_length=inputLength,
            mask_zero=True,
            name='encoderEmbedding'
        )

        self.lstm = LSTM(
            self.enc_units,
            return_sequences=True,
            name="encoderLSTM"
        )

        self.time_step_slicer = GetLastTimestepLayer(
            name="encoderLastStateExtractor")

    def call(self, x):
        # Get the embeddings for the input
        x = self.embedding(x)

        # Run in trough the LSTM
        encoder_hidden = self.lstm(x)

        encoder_last_state = self.time_step_slicer(encoder_hidden)

        return encoder_hidden, encoder_last_state
