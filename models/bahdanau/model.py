from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from tensorflow.keras.models import Model

from models.bahdanau.encoder import Encoder
from models.bahdanau.decoder import Decoder

def createModel(inputVocabSize, outputVocabSize, inputLength, outputLength):
    embeddingDims = 64
    lstmUnits = 64

    print(f"inputVocabSize {inputVocabSize}")
    print(f"outputVocabSize {outputVocabSize}")
    print(f"inputLength {inputLength}")
    print(f"outputLength {outputLength}")

    # Encoder
    encoderEmbeddingInput = Input(
        shape=(inputLength,), name='embeddingEncoderInput')

        # Decoder
    decoderEmbeddingInput = Input(
        shape=(outputLength,), name='embeddingDecoderInput')

    ### Encoder 
    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits)
    # hidden = encoder.initialize_hidden_state()
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(encoderEmbeddingInput)

    decoder = Decoder(outputVocabSize, embeddingDims, lstmUnits)
    decoderOutput = decoder(decoderEmbeddingInput, [encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)
    
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
