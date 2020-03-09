from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from tensorflow.keras.models import Model

from models.luong.encoder import Encoder
from models.luong.decoder import Decoder


def createModel(inputVocabSize, outputVocabSize, inputLength, outputLength, embeddingDims, lstmUnits):

    # Encoder
    encoder_input = Input(
        shape=(inputLength,), name='embeddingEncoderInput')

    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits)

    encoder_output, dec_init_state = encoder(encoder_input)

    # Decoder
    decoder_input = Input(
        shape=(outputLength,), name='embeddingDecoderInput')
    decoder = Decoder(outputVocabSize, embeddingDims, lstmUnits)

    decoderCombinedContext = decoder(decoder_input, dec_init_state, encoder_output)

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
        inputs=[encoder_input, decoder_input],
        outputs=outputGenerator
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam"
    )

    return model
