from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from models.pointer_network.encoder import Encoder
from models.pointer_network.decoder import Decoder

def createModel(inputVocabSize, outputVocabSize, inputLength, outputLength, embeddingDims, lstmUnits):
    # Encoder
    encoderEmbeddingInput = Input(shape=(inputLength,), name='embeddingEncoderInput')
    
    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits)
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(
        encoderEmbeddingInput)

    # Decoder
    decoderEmbeddingInput = Input(shape=(outputLength,), name='embeddingDecoderInput')

    decoder = Decoder(outputVocabSize, embeddingDims, lstmUnits)
    
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