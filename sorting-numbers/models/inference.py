import sys
sys.path.append(".")
import numpy as np

def runSeq2SeqInference(model, encoderInput, vocab_size, input_length, max_value):
    EOS_CODE = max_value + 0 # max_value is not included in range(max_value)
    SOS_CODE = max_value + 1 # next element

    # Init Decoder's input to zeros
    decoderInput = np.zeros((1, input_length), dtype="float32")
    # Add start-of-sequence SOS into decoder
    decoderInput[0,0] = SOS_CODE

    for i in range(1, input_length):
        # Make a predition
        predictOut = model.predict([encoderInput, decoderInput])
        # Get the result
        output = predictOut.argmax(2)[0, i-1]
        # Append it to the decoder's input
        decoderInput[0, i] = output

    finalPrediction = model.predict([encoderInput, decoderInput])
    lastPointer = finalPrediction.argmax(2)[0, input_length-1]

    encoderInput = encoderInput.numpy().astype("int16")[0]
    decoderInput = decoderInput.astype("int16")

    output = []
    for i in range(1, decoderInput.shape[1]):
        pointer = decoderInput[0, i]
        output.append(encoderInput[pointer])

    output.append(encoderInput[lastPointer])

    # print(output)

    return output[:-1]


if __name__ == "__main__":
    pass