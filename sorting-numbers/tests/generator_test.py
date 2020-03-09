import sys
sys.path.append('./sorting-numbers')
from dataset.generator import ArtificialDataset
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
        dataset = ArtificialDataset(4)

        count = 0
        for _ in dataset:
             count += 1
        
        self.assertEqual(count, 4)
        
    def test_tensorShapes(self):
        dataset = ArtificialDataset(4, 11)

        enc_in = []
        dec_in = []
        for d in dataset:
             enc_in.append(d['enc_in'])
             dec_in.append(d['dec_in'])
        
        self.assertEqual(len(enc_in), 4)
        self.assertEqual(enc_in[0].shape, [11])

        self.assertEqual(len(dec_in), 4)
        self.assertEqual(dec_in[0].shape, [11])
    
    def test_tensorContents(self):
        dataset = ArtificialDataset(4, 10, 100)

        enc_in = []
        dec_in = []
        dec_out = []
        for d in dataset:
             enc_in.append(d['enc_in'])
             dec_in.append(d['dec_in'])
             dec_out.append(d['dec_out'])
        
        
        # Sort encoder's input
        sortedInput = np.sort(enc_in[0].numpy())

        # Convert encoder's input into one-hot
        in_array = tf.one_hot(tf.cast(sortedInput, tf.int32), 101)
        dec_array = dec_out[0].numpy()

        # Compare encoder's input with to decoder's output
        self.assertTrue(np.all(in_array == dec_array))

        # Decoder input should start with START CODE (101)
        out_array = dec_in[0].numpy()
        self.assertEqual(out_array[0], 101)
        
        


if __name__ == '__main__':
    unittest.main()