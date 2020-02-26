from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from models.luong.last_time_step_layer import GetLastTimestepLayer

def createModel(inputVocabSize, outputVocabSize, inputLength, outputLength):
    embeddingDims = 64
    lstmUnits = 64

    print(f"inputVocabSize {inputVocabSize}")
    print(f"outputVocabSize {outputVocabSize}")
    print(f"inputLength {inputLength}")
    print(f"outputLength {outputLength}")

    encoderEmbeddingInput = Input(shape=(inputLength,), name='embeddingEncoderInput')

    encoderEmbeddingOutput = Embedding(
        inputVocabSize,
        embeddingDims,
        input_length = inputLength,
        mask_zero = True,
        name = 'encoderEmbedding'
    )(encoderEmbeddingInput)

    encoderLSTMOutput = LSTM(
        lstmUnits,
        return_sequences = True,
        name="encoderLSTM"
    )(encoderEmbeddingOutput)

    encoderLastState = GetLastTimestepLayer(encoderLSTMOutput)

    decoderEmbeddingInput = Input(shape=(outputLength,), name='embeddingDecoderInput')

    decoderEmbeddingOutput = Embedding(
        outputVocabSize,
        embeddingDims,
        input_length = outputLength,
        mask_zero = True,
        name = 'decoderEmbedding'
    )(decoderEmbeddingInput)

    decoderLSTMOutput = LSTM(
        lstmUnits,
        return_sequences = True,
        name="decoderLSMT"
    )(decoderEmbeddingOutput, initial_state=[encoderLastState, encoderLastState])

    outputGenerator = TimeDistributed(
        Dense(outputVocabSize, activation = "softmax"),
        name = "timeDistributedSoftmax"
    )(decoderLSTMOutput)

    model = Model(
        inputs = [encoderEmbeddingInput, decoderEmbeddingInput],
        outputs = outputGenerator
    )

    model.compile(
        loss = "categorical_crossentropy",
        optimizer = "Adam"
    )

    return model