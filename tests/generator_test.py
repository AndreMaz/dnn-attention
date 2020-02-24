import unittest
import sys
sys.path.append('.')
from dataset.generator import first3Letters, MONTH_NAMES_FULL


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

    def test_first3letters(self):
        result = first3Letters(MONTH_NAMES_FULL)
        self.assertEqual(len(result), 12)

if __name__ == '__main__':
    unittest.main()
