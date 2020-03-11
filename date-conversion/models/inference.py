import sys
sys.path.append(".")
from dataset.date_format import encodeInputDateStrings, OUTPUT_LENGTH, START_CODE, OUTPUT_VOCAB
import numpy as np
import tensorflow as tf

def runSeq2SeqInference(modelName, model, inputStr):
    # Encode encoder's input date
    encoderInput = encodeInputDateStrings([inputStr])

    # Init Decoder's input to zeros
    decoderInput = np.zeros((1, OUTPUT_LENGTH), dtype="float32")
    # Add start code into decoder
    decoderInput[0,0] = START_CODE

    for i in range(1, OUTPUT_LENGTH):
        # Make a predition
        if (modelName != 'seq2seq'):
            predictOut, _ = model.predict([encoderInput, decoderInput])
        else:
            predictOut = model.predict([encoderInput, decoderInput])
        # Get the result
        output = predictOut.argmax(2)[0, i-1]
        # Append it to the decoder's input
        decoderInput[0, i] = output

    # Make the prediction of the last char
    if (modelName != 'seq2seq'):
        finalPredictOut, attention_weights = model.predict([encoderInput, decoderInput])
    else:
        finalPredictOut = model.predict([encoderInput, decoderInput])

    decoderFinalOutput = finalPredictOut.argmax(2)[0, OUTPUT_LENGTH-1]

    # Map the indices to chars
    outputStr = ""
    for i in range(1, decoderInput.shape[1]):
        outputStr += OUTPUT_VOCAB[int(decoderInput[0, i])]

    outputStr += OUTPUT_VOCAB[int(decoderFinalOutput)]

    if (modelName != 'seq2seq'):
        return outputStr, attention_weights
    else:
        return outputStr


if __name__ == "__main__":
    pass