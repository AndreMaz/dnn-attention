import tensorflow as tf
import random

def generateEncoderInput(num_samples, size, upperLimit):
    lowerLimit = 0
    START_CODE = upperLimit + 1

    encoderInputs = []
    decoderInputs = []
    decoderOutputs = []

    for _ in range(num_samples):

        enc_input_list = []
        for _ in range(size):
            enc_input_list.append(random.randint(lowerLimit, upperLimit))
        
        # Decoder's input is equal to the sorted encoder's input
        dec_input_list = list(enc_input_list)
        dec_input_list.sort()

        # Decoder should produce sorted list
        dec_output_list = list(dec_input_list)

        # Remove last element from decoder's input
        
        # Add Start Code
        dec_input_list = [START_CODE] + dec_input_list[1:]

        encoderInputs.append(tf.convert_to_tensor(enc_input_list, dtype="float32"))
        
        decoderInputs.append(tf.convert_to_tensor(dec_input_list, dtype="float32"))

        decoderOutputs.append(tf.one_hot(tf.convert_to_tensor(dec_output_list, dtype="int32"), upperLimit + 1)) # +1 for START CODE)

    return tf.convert_to_tensor(encoderInputs), tf.convert_to_tensor(decoderInputs), tf.convert_to_tensor(decoderOutputs)
    # return 1