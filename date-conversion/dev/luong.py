import sys
sys.path.append('.')
from tensorflow.keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from models.luong.last_time_step_layer import GetLastTimestepLayer
from dataset.generator import encodeInputDateStrings

inputVocabSize = 35
inputLength = 12
outputVocabSize = 13
outputLength = 10

def runner():
    embeddingDims = 64
    lstmUnits = 64

    encoderEmbeddingInput = encodeInputDateStrings(["1.8.2020"])

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

    layer = GetLastTimestepLayer()
    encoderLastState = layer(encoderLSTMOutput)
    
    print(encoderLastState.shape)

    return 1


if __name__ == "__main__":
    runner()