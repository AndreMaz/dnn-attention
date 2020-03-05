from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.pointer_network.encoder import Encoder
from models.pointer_network.decoder import Decoder

def createModel(vocab_size, seq_length, embedding_dims, lstm_units):
    # Encoder
    encoderEmbeddingInput = Input(shape=(seq_length,), name='embeddingEncoderInput')
    
    encoder = Encoder(vocab_size, embedding_dims, lstm_units)
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(
        encoderEmbeddingInput)

    # Decoder
    decoderEmbeddingInput = Input(shape=(seq_length,), name='embeddingDecoderInput')

    decoder = Decoder(vocab_size, embedding_dims, lstm_units)

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