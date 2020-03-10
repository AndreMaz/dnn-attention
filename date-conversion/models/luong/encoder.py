from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from tensorflow.keras import Model
from tensorflow import zeros
from models.luong.last_time_step_layer import GetLastTimestepLayer


class Encoder(Model):
    def __init__(self, inputVocabSize, embeddingDims, lstmUnits, inputLength):
        super(Encoder, self).__init__()
        # self.enc_units = enc_units

        self.embedding = Embedding(
            inputVocabSize,
            embeddingDims,
            input_length=inputLength,
            mask_zero=True,
            name='encoderEmbedding'
        )

        self.lstm = LSTM(
            lstmUnits,
            return_sequences=True,
            name="encoderLSTM"
        )

        self.time_step_slicer = GetLastTimestepLayer(
            name="encoderLastStateExtractor")

    def call(self, x):
        # Get the embeddings for the input
        encoderEmbeddingOutput = self.embedding(x)

        # Run in trough the LSTM
        encoderLSTMOutput = self.lstm(encoderEmbeddingOutput)

        encoderLastState = self.time_step_slicer(encoderLSTMOutput)

        return encoderLSTMOutput, encoderLastState
