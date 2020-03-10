from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from tensorflow.keras.models import Model

from models.luong.encoder import Encoder
from models.luong.decoder import Decoder


def createModel(inputVocabSize, outputVocabSize, inputLength, outputLength, embeddingDims, lstmUnits):

    # Encoder
    encoderEmbeddingOutput = Input(
        shape=(inputLength,), name='embeddingEncoderInput')

    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits, inputLength)
    encoderLSTMOutput, encoderLastState = encoder(encoderEmbeddingOutput)

    # Decoder
    decoderEmbeddingInput = Input(
        shape=(outputLength,), name='embeddingDecoderInput')

    decoder = Decoder(outputVocabSize, embeddingDims, lstmUnits, outputLength)
    decoderCombinedContext, attention_weights = decoder(decoderEmbeddingInput, encoderLSTMOutput, encoderLastState)

    # Prediction Layers
    outputGeneratorTanh = TimeDistributed(
        Dense(lstmUnits, activation="tanh"),
        name="timeDistributedTanh"
    )(decoderCombinedContext)

    outputGenerator = TimeDistributed(
        Dense(outputVocabSize, activation="softmax"),
        name="timeDistributedSoftmax"
    )(outputGeneratorTanh)

    model = Model(
        inputs=[encoderEmbeddingOutput, decoderEmbeddingInput],
        outputs=outputGenerator
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam"
    )

    return model
