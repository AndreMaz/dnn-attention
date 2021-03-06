from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dot, Activation, Concatenate
from tensorflow.keras.models import Model

from models.luong.encoder import Encoder
from models.luong.decoder import Decoder
from models.luong.attention import LuongAttention
from models.luong.last_time_step_layer import GetLastTimestepLayer


def createModel(inputVocabSize, outputVocabSize, inputLength, outputLength, embeddingDims, lstmUnits):

    # Encoder
    encoderEmbeddingInput = Input(
        shape=(inputLength,), name='embeddingEncoderInput')

    # encoderEmbeddingOutput = Embedding(
    #     inputVocabSize,
    #     embeddingDims,
    #     input_length=inputLength,
    #     mask_zero=True,
    #     name='encoderEmbedding'
    # )(encoderEmbeddingInput)

    # encoderLSTMOutput = LSTM(
    #     lstmUnits,
    #     return_sequences=True,
    #     name="encoderLSTM"
    # )(encoderEmbeddingOutput)

    # # Get last hidden state
    # encoderLastState = GetLastTimestepLayer(
    #     name="encoderLastStateExtractor")(encoderLSTMOutput)


    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits, inputLength)
    encoderLSTMOutput, encoderLastState = encoder(encoderEmbeddingInput)

    # Decoder
    decoderEmbeddingInput = Input(
        shape=(outputLength,), name='embeddingDecoderInput')

    # decoderEmbeddingOutput = Embedding(
    #     outputVocabSize,
    #     embeddingDims,
    #     input_length=outputLength,
    #     mask_zero=True,
    #     name='decoderEmbedding'
    # )(decoderEmbeddingInput)

    # decoderLSTMOutput = LSTM(
    #     lstmUnits,
    #     return_sequences=True,
    #     name="decoderLSMT"
    # )(decoderEmbeddingOutput, initial_state=[encoderLastState, encoderLastState])

    
    # ##### ATTENTION #####
    # # attention = Dot((2, 2), name="attentionDot")(
    # #     [decoderLSTMOutput, encoderLSTMOutput])

    # # attention = Activation("softmax", name="attentionSoftMax")(attention)

    # # context = Dot((2, 1), name="context")([attention, encoderLSTMOutput])

    # att = LuongAttention()
    # context, attention = att(decoderLSTMOutput, encoderLSTMOutput)
    # ##### ATTENTION #####

    # decoderCombinedContext = Concatenate(
    #     name="combinedContext")([context, decoderLSTMOutput])
    
    decoder = Decoder(outputVocabSize, embeddingDims, lstmUnits, outputLength)
    decoderCombinedContext, _ = decoder(decoderEmbeddingInput, encoderLSTMOutput, encoderLastState)

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
        inputs=[encoderEmbeddingInput, decoderEmbeddingInput],
        outputs=outputGenerator
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="Adam"
    )

    return model

