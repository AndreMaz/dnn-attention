import tensorflow as tf
import numpy as np
import random

def generateEncoderInput(num_samples, sample_length, max_value, vocab_size):
    min_value = 1
    START_CODE = 0


    encoderInputs = []
    decoderInputs = []
    decoderOutputs = []

    # Generate sequence of possible numbers
    num_sequence = []
    for i in range(min_value, max_value + 1): # +1 to include the upper limit 
        num_sequence.append(i)

    for _ in range(num_samples):
        # Shuffle the numbers list
        random.shuffle(num_sequence)
        
        # Get a slice
        enc_input_list = list(num_sequence[:sample_length])

        # Decoder's input is equal to the sorted encoder's input BUT:
        # - it's sorted
        # - it has a START_CODE at the beggining
        dec_input_list = list(enc_input_list)
        dec_input_list.sort()

        
        dec_output_list = list(enc_input_list)

        # Add the START CODE
        dec_input_list = [START_CODE] + dec_input_list[:-1]

        encoderInputs.append(tf.convert_to_tensor(enc_input_list, dtype="float32"))
        
        decoderInputs.append(tf.convert_to_tensor(dec_input_list, dtype="float32"))
        print(enc_input_list)

        dec_out_enc = np.zeros((sample_length, vocab_size), dtype='int32')
        for i in range(min_value, max_value + 1):
        # for index,value in enumerate(enc_input_list):
            pos = enc_input_list.index(i)
            # i-1 because of the range in the loop
            # pos+1 because 0 is the START CODE
            dec_out_enc[i-1][pos+1] = 1
        # print(x)

        decoderOutputs.append(dec_out_enc)

    return tf.convert_to_tensor(encoderInputs), tf.convert_to_tensor(decoderInputs), tf.convert_to_tensor(decoderOutputs)
    # return 1