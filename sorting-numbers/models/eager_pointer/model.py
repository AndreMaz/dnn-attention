from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.pointer_network.encoder import Encoder
from models.pointer_network.decoder import Decoder

from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras import Model


class EagerModel(Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(EagerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.lstm_units)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, self.lstm_units)

    def call(self, encoder_input, decoder_input):
        encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = self.encoder(
        encoder_input)

        decoderOutput = self.decoder(decoder_input, [
                            encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)

        return decoderOutput