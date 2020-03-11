import sys
sys.path.append('./sorting-numbers')
import tensorflow as tf
from models.pointer_masking.encoder import Encoder
from models.pointer_masking.decoder import Decoder
from dataset.generator import generateDataset

embeddingDims = 64
lstmUnits = 64

num_samples = 2
sample_length = 10
max_value = 10

vocab_size = max_value + 2 # +2 for SOS and EOS
input_length = sample_length + 1 # For special chars at the beggining

def runner():
    encoderEmbeddingInput, decoderEmbeddingInput, trainDecoderOutput = generateDataset(num_samples, sample_length, max_value, vocab_size)

    print(encoderEmbeddingInput)
    print(decoderEmbeddingInput)
    print(trainDecoderOutput)

    encoder = Encoder(vocab_size, embeddingDims, lstmUnits)
    encoderHiddenStates, encoderLastHiddenState, encoderLastCarryState = encoder(encoderEmbeddingInput)

    decoder = Decoder(input_length, vocab_size, embeddingDims, lstmUnits)
    decoderOutput = decoder(decoderEmbeddingInput, [encoderLastHiddenState, encoderLastCarryState], encoderHiddenStates)

    return 1


if __name__ == "__main__":
    runner()