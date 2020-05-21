import sys
sys.path.append(".")
import numpy as np

def runSeq2SeqInference(model, encoderInput, vocab_size, input_length, max_value, SOS_CODE, EOS_CODE, eager = False):
    # Init Decoder's input to zeros
    decoderInput = np.zeros((1, input_length), dtype="float32")
    # Add start-of-sequence SOS into decoder
    decoderInput[0,0] = SOS_CODE

    for i in range(1, input_length):
        # Make a predition
        if not eager:
            predictOut = model.predict([encoderInput, decoderInput])
            pointer = predictOut.argmax(2)[0, i-1]
        else:
            predictOut = model(encoderInput, decoderInput)
            # Get the pointer
            pointer = predictOut.numpy().argmax(2)[0, i-1]

        # Get the value that pointer points to
        valuePointed = encoderInput[0, pointer]
        decoderInput[0, i] = valuePointed

    # Final pointer given the full decoder's sequence as input
    # If model is trained well it should point to EOS symbol
    if not eager:
        finalPrediction = model.predict([encoderInput, decoderInput])
        finalPointer = finalPrediction.argmax(2)[0, i-1]
    else:
        finalPrediction = model(encoderInput, decoderInput)
        finalPointer = finalPrediction.numpy().argmax(2)[0, i-1]

    finalValue = encoderInput[0, finalPointer]

    # Return only the number sequence
    # 1st element is SOS
    return list(decoderInput[0].astype("int16"))[1:], finalPrediction


if __name__ == "__main__":
    pass