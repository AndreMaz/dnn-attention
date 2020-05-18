import tensorflow as tf
import numpy as np
import random

def generateDataset(num_samples, sample_length, max_value, vocab_size):
    min_value = 1
    EOS_CODE = max_value + 1 # max_value is not included in range(max_value)
    SOS_CODE = max_value + 2 # next element
    
    encoderInputs = []
    decoderInputs = []
    decoderOutputs = []

    # Generate sequence of possible numbers
    num_sequence = []
    for i in range(min_value, max_value + 1):
        num_sequence.append(i)

    for _ in range(num_samples):
        # Shuffle the numbers list
        random.shuffle(num_sequence)
        
        # Get a slice of shuffled numbers and add EOS_CODE
        enc_input_list = [EOS_CODE] + list(num_sequence[:sample_length])

        # Decoder's input is equal to the encoder's input BUT:
        # - it's sorted
        # - it has a START_CODE at the beggining
        dec_input_list = list(enc_input_list[1:])
        dec_input_list.sort()

        # Add the SOS to decoder's input
        dec_input_list = [SOS_CODE] + dec_input_list

        encoderInputs.append(tf.convert_to_tensor(enc_input_list, dtype="float32"))
        
        decoderInputs.append(tf.convert_to_tensor(dec_input_list, dtype="float32"))

        # Decoder's output
        dec_out_enc = np.zeros((sample_length + 1, sample_length + 1), dtype='int32')
        
        # Sort the input list
        sorted_sequence = list(enc_input_list[1:])
        sorted_sequence.sort()
        sorted_sequence = sorted_sequence + [EOS_CODE]
        for index, value in enumerate(sorted_sequence):
            # Get the index from "value" in unsorted input
            pos = enc_input_list.index(value)
            # Set the one-hot
            dec_out_enc[index][pos] = 1

        decoderOutputs.append(dec_out_enc)

    return tf.convert_to_tensor(encoderInputs), tf.convert_to_tensor(decoderInputs), tf.convert_to_tensor(decoderOutputs)