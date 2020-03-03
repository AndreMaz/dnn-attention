import sys
sys.path.append('./sorting-numbers')
from dataset.generator import ArtificialDataset
import unittest

class TestArtificialDataset(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    # def test_generate(self):
    #     dataset = ArtificialDataset(4)
        
    


if __name__ == '__main__':
    unittest.main()