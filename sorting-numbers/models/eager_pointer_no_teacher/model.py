from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.eager_pointer_no_teacher.encoder import Encoder
from models.eager_pointer_no_teacher.decoder import Decoder

from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras import Model


class EagerModel(Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, SOS_CODE):
        super(EagerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.SOS_CODE = SOS_CODE

        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.lstm_units)
        self.decoder = Decoder(self.SOS_CODE, self.vocab_size, self.embedding_dim, self.lstm_units)

    def call(self, encoder_input):
        encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = self.encoder(
        encoder_input)

        decoderOutput = self.decoder([encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)

        return decoderOutput