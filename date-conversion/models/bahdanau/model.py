from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from tensorflow.keras.models import Model

from models.bahdanau.encoder import Encoder
from models.bahdanau.decoder import Decoder


def createModel(inputVocabSize, outputVocabSize, inputLength, outputLength, embeddingDims, lstmUnits):

    # Encoder
    encoderEmbeddingInput = Input(
        shape=(inputLength,), name='embeddingEncoderInput')

    # Encoder
    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits)

    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(
        encoderEmbeddingInput)

    # Decoder
    decoderEmbeddingInput = Input(
        shape=(outputLength,), name='embeddingDecoderInput')

    decoder = Decoder(outputVocabSize, embeddingDims, lstmUnits)
    # Ignore attention at this point
    # We only need it during testing
    decoderOutput, _ = decoder(decoderEmbeddingInput, [
                            encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)

    # Prediction Layers
    outputGeneratorTanh = TimeDistributed(
        Dense(lstmUnits, activation="tanh"),
        name="timeDistributedTanh"
    )(decoderOutput)

    outputGenerator = TimeDistributed(
        Dense(outputVocabSize, activation="softmax"),
        name="timeDistributedSoftmax"
    )(outputGeneratorTanh)

    model = Model(
        inputs=[encoderEmbeddingInput, decoderEmbeddingInput],
        outputs=outputGenerator
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam"
    )

    return model
