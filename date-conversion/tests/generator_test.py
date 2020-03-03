import sys
sys.path.append('./date-conversion')
from dataset import generator
import unittest

class TestStringMethods(unittest.TestCase):

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

    def test_generate(self):
        result = generator.generateOrderedDates('1950-01-01', '1950-02-01')
        self.assertEqual(len(result), 32)


if __name__ == '__main__':
    unittest.main()
