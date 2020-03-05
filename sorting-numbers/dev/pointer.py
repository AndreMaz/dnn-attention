import sys
sys.path.append('./sorting-numbers')
import tensorflow as tf
from models.pointer_network.encoder import Encoder
from models.pointer_network.decoder import Decoder
from dataset.generator import generateDataset

embeddingDims = 64
lstmUnits = 64

num_samples = 1
sample_length = 10
max_value = 10

input_length = sample_length + 1 # For special chars
vocab_size = max_value + 2 # +2 for SOS and EOS


def runner():
    encoderEmbeddingInput, decoderEmbeddingInput, trainDecoderOutput = generateDataset(num_samples, sample_length, max_value, vocab_size)

    print(encoderEmbeddingInput)
    print(decoderEmbeddingInput)
    print(trainDecoderOutput)

    encoder = Encoder(vocab_size, embeddingDims, lstmUnits)
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(encoderEmbeddingInput)

    decoder = Decoder(vocab_size, embeddingDims, lstmUnits)
    decoderOutput = decoder(decoderEmbeddingInput, [encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)

    print(decoderOutput)
    return 1


if __name__ == "__main__":
    runner()