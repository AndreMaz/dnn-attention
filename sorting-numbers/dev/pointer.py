import sys
sys.path.append('./sorting-numbers')
import tensorflow as tf
from models.pointer_network.encoder import Encoder
from models.pointer_network.decoder import Decoder

inputVocabSize = 10 + 1 # 10 for number + 1 for START CODE
embeddingDims = 64
lstmUnits = 64

def runner():
    # Encoder Input
    encoderEmbeddingInput = tf.convert_to_tensor([[0, 8, 9, 7, 5, 4, 6, 2, 3, 1], [0, 8, 9, 7, 5, 4, 6, 2, 3, 1]], dtype="float32")
    
    decoder_input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    STOP_SYMBOL = 10 
    # Shifted Decoder Input
    decoderEmbeddingInput = tf.convert_to_tensor( [[STOP_SYMBOL, 1, 2, 3, 4, 5, 6, 7, 8, 9], [STOP_SYMBOL, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype="float32")

    decoder_output = tf.one_hot(decoder_input, len(decoder_input) + 1)
    print(decoder_output)

    encoder = Encoder(inputVocabSize, embeddingDims, lstmUnits)
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(encoderEmbeddingInput)

    decoder = Decoder(inputVocabSize, embeddingDims, lstmUnits)
    res = decoder(decoderEmbeddingInput, [encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)

    print(res)
    return 1


if __name__ == "__main__":
    runner()