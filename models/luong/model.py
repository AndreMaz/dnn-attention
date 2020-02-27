from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from tensorflow.keras.models import Model
from models.luong.last_time_step_layer import GetLastTimestepLayer


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

    encoderEmbeddingOutput = Embedding(
        inputVocabSize,
        embeddingDims,
        input_length=inputLength,
        mask_zero=True,
        name='encoderEmbedding'
    )(encoderEmbeddingInput)

    encoderLSTMOutput = LSTM(
        lstmUnits,
        return_sequences=True,
        name="encoderLSTM"
    )(encoderEmbeddingOutput)

    # Get last hidden state
    slicerLayer = GetLastTimestepLayer()
    encoderLastState = slicerLayer(encoderLSTMOutput)

    # Decoder
    decoderEmbeddingInput = Input(
        shape=(outputLength,), name='embeddingDecoderInput')

    decoderEmbeddingOutput = Embedding(
        outputVocabSize,
        embeddingDims,
        input_length=outputLength,
        mask_zero=True,
        name='decoderEmbedding'
    )(decoderEmbeddingInput)

    decoderLSTMOutput = LSTM(
        lstmUnits,
        return_sequences=True,
        name="decoderLSMT"
    )(decoderEmbeddingOutput, initial_state=[encoderLastState, encoderLastState])

    attentionDotLayer = Dot((2, 2), name="attentionDot")
    attention = attentionDotLayer([decoderLSTMOutput, encoderLSTMOutput])

    activationLayer = Activation("softmax", name="attentionSoftMax")
    attention = activationLayer(attention)

    contextDotLayer = Dot((2, 1), name="context")
    context = contextDotLayer([attention, encoderLSTMOutput])

    concatenateLayer = Concatenate(name="combinedContext")
    decoderCombinedContext = concatenateLayer([context, decoderLSTMOutput])

    outputGenerator = TimeDistributed(
        Dense(lstmUnits, activation="tanh",),
        name="timeDistributedTanh"
    )(decoderCombinedContext)

    outputGenerator = TimeDistributed(
        Dense(outputVocabSize, activation="softmax"),
        name="timeDistributedSoftmax"
    )(outputGenerator)

    model = Model(
        inputs=[encoderEmbeddingInput, decoderEmbeddingInput],
        outputs=outputGenerator
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam"
    )

    return model
