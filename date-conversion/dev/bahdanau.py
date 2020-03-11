import sys
sys.path.append('./date-conversion')
from models.bahdanau.encoder import Encoder
from models.bahdanau.decoder import Decoder
from dataset.generator import encodeInputDateStrings, encodeOutputDateStrings
from tensorflow.keras.layers import Embedding, Dense, TimeDistributed

inputVocabSize = 35
inputLength = 12
outputVocabSize = 13
outputLength = 10
batchSize = 2

def runner():
    embeddingDims = 64
    lstmUnits = 64

    encoderEmbeddingInput = encodeInputDateStrings(["1.8.2020", "1.8.2020"])

    ### Encoder 
    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits)
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(encoderEmbeddingInput)

    ### Decoder
    decoderEmbeddingInput = encodeOutputDateStrings(["2020-08-01", "2020-08-01"])
    decoder = Decoder(outputVocabSize, embeddingDims, lstmUnits)
    decoderOutput, attention_weights = decoder(decoderEmbeddingInput, [encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)
    
    outputGeneratorTanh = TimeDistributed(
        Dense(lstmUnits, activation="tanh"),
        name="timeDistributedTanh"
    )(decoderOutput)

    outputGenerator = TimeDistributed(
        Dense(outputVocabSize, activation="softmax"),
        name="timeDistributedSoftmax"
    )(outputGeneratorTanh)

    # rnn = customRNN(outputVocabSize, embeddingDims, lstmUnits, batchSize)
    # res = rnn(embeddingOut)
    print(outputGenerator)


    return 1

if __name__ == "__main__":
    runner()