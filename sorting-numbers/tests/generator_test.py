import sys
sys.path.append('./sorting-numbers')
from dataset.generator import generateDataset
import unittest
import numpy as np
import tensorflow as tf

class TestArtificialDataset(unittest.TestCase):
    # def test_upper(self):
    #    self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #    self.assertTrue('FOO'.isupper())
    #    self.assertFalse('Foo'.isupper())

    def test_generate(self):
        num_samples = 2 # number of samples to generate
        sample_length = 10 # Length of input sequence
        max_value = 100 # Upper bound (range.random()) to generate a number
        vocab_size = max_value + 2 # +2 for SOS and EOS

        trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateDataset(num_samples, sample_length, max_value, vocab_size)

        self.assertEqual(len(trainEncoderInput), 2)
        self.assertEqual(len(trainDecoderInput), 2)
        self.assertEqual(len(trainDecoderOutput), 2)
        
    def test_tensorShapes(self):
        num_samples = 2 # number of samples to generate
        sample_length = 10 # Length of input sequence
        max_value = 100 # Upper bound (range.random()) to generate a number
        vocab_size = max_value + 2 # +2 for SOS and EOS

        trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateDataset(num_samples, sample_length, max_value, vocab_size)
        
        self.assertEqual(trainEncoderInput[0].shape, [11])

        self.assertEqual(trainDecoderInput[0].shape, [11])

        self.assertEqual(trainDecoderOutput[0].shape, [11, 11])
    
    def test_tensorContents(self):
        num_samples = 2 # number of samples to generate
        sample_length = 10 # Length of input sequence
        max_value = 100 # Upper bound (range.random()) to generate a number
        vocab_size = max_value + 2 # +2 for SOS and EOS

        trainEncoderInput, trainDecoderInput, trainDecoderOutput = generateDataset(num_samples, sample_length, max_value, vocab_size)
        
        encoderInput = trainEncoderInput[0].numpy()
        decoderInput = trainDecoderInput[0].numpy()

        # Sorted Encoder input should be equal to Decoder's Input
        # The SOS and EOS should are not considered
        self.assertTrue(np.all(np.sort(encoderInput[1:]) == decoderInput[1:]))

        
        


if __name__ == '__main__':
    unittest.main()