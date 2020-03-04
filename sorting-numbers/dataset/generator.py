
import tensorflow as tf
import random

class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples, size, upperLimit):
        # Opening the file
        lowerLimit = 0
        
        START_CODE = upperLimit + 1

        entry = {}

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

            input = tf.convert_to_tensor(enc_input_list, dtype="float32")
            # input = tf.random.uniform((1, size), minval = lowerLimit, maxval = upperLimit, dtype="int32")
            entry['enc_in'] = input
            
            entry['dec_in'] = tf.convert_to_tensor(dec_input_list, dtype="float32")

            entry['dec_out'] = tf.one_hot(tf.convert_to_tensor(dec_output_list, dtype="int32"), upperLimit + 1) # +1 for START CODE

            yield (entry)
    
    def __new__(self, num_samples=3, size = 10, upperLimit = 100):
        return tf.data.Dataset.from_generator(
            self._generator,
            output_types=({"enc_in": tf.dtypes.float32, "dec_in": tf.dtypes.float32, "dec_out": tf.dtypes.int32, }),
            output_shapes=({"enc_in": (size, ), "dec_in": (size), "dec_out": (size, upperLimit+1)}),
            args=(num_samples, size, upperLimit)
        )