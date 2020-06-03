from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.pointer_masking.encoder import Encoder
from models.pointer_masking.decoder import Decoder, DecoderNoTrainer

def createModel(vocab_size, input_length, embedding_dims, lstm_units):
    # Encoder
    encoderEmbeddingInput = Input(shape=(input_length,), name='embeddingEncoderInput')
    
    encoder = Encoder(vocab_size, embedding_dims, lstm_units)
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(
        encoderEmbeddingInput)

    # Decoder
    decoderEmbeddingInput = Input(shape=(input_length,), name='embeddingDecoderInput')

    decoder = Decoder(input_length, vocab_size, embedding_dims, lstm_units)

    decoderOutput = decoder(decoderEmbeddingInput, [
                            encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)


    model = Model(
        inputs = [encoderEmbeddingInput, decoderEmbeddingInput],
        outputs = decoderOutput
    )

    model.compile(
        loss = "categorical_crossentropy",
        optimizer = "Adam"
    )

    return model


class EagerModelNoTrainer(Model):
    def __init__(self, input_length, vocab_size, embedding_dim, lstm_units, SOS_CODE):
        super(EagerModelNoTrainer, self).__init__()
        self.input_length = input_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.SOS_CODE = SOS_CODE

        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.lstm_units)
        self.decoder = DecoderNoTrainer(self.input_length, self.vocab_size, self.embedding_dim, self.lstm_units, self.SOS_CODE)

    def call(self, encoder_input):
        encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = self.encoder(
        encoder_input)

        decoderOutput = self.decoder([encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates, encoder_input)

        return decoderOutput