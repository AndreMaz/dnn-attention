import unittest
import sys
sys.path.append('.')
from dataset import generator


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
        result = generator.generate('1950-01-01', '1950-02-01')
        self.assertEqual(len(result), 32)

    def test_first3letters(self):
        result = generator.first3Letters(generator.MONTH_NAMES_FULL)
        self.assertEqual(len(result), 12)
        self.assertEqual(result[0], "JAN")
        self.assertEqual(result[1], "FEB")

    def test_uniqueMonthLetters(self):
        monthList = generator.first3Letters(generator.MONTH_NAMES_FULL)
        result = generator.uniqueMonthLetters(monthList)
        self.assertEqual(result, "ABCDEFGJLMNOPRSTUVY")

    def test_InputVocab(self):
        self.assertEqual(len(generator.INPUT_VOCAB), 35)

    def test_OutputVocab(self):
        self.assertEqual(len(generator.OUTPUT_VOCAB), 13)

    def test_toTwoDigitString(self):
        self.assertEqual(generator.toTwoDigitString(2), "02")
        self.assertEqual(generator.toTwoDigitString(11), "11")

    def test_dateTupleToDDMMMYYYY(self):
        dateTuple = [2020,12,30]
        self.assertEqual(generator.dateTupleToDDMMMYYYY(dateTuple), "30DEC2020")

    def test_dateTupleToMMSlashDDSlashYYYY(self):
        dateTuple = [2020,12,30]
        self.assertEqual(generator.dateTupleToMMSlashDDSlashYYYY(dateTuple), "12/30/2020")

    def test_dateTupleToMSlashDSlashYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToMSlashDSlashYYYY(dateTuple), "8/1/2020")

    def test_dateTupleToMMSlashDDSlashYY(self):
        dateTuple = [2021,8,30]
        self.assertEqual(generator.dateTupleToMMSlashDDSlashYY(dateTuple), "08/30/21")

    def test_dateTupleToMSlashDSlashYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToMSlashDSlashYY(dateTuple), "8/1/20")

    def test_dateTupleToMMDDYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToMMDDYY(dateTuple), "080120")

    def test_dateTupleToMMMSpaceDDSpaceYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToMMMSpaceDDSpaceYY(dateTuple), "AUG 01 20")

    def test_dateTupleToMMMSpaceDDSpaceYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToMMMSpaceDDSpaceYYYY(dateTuple), "AUG 01 2020")

    def test_dateTupleToMMMSpaceDDCommaSpaceYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToMMMSpaceDDCommaSpaceYY(dateTuple), "AUG 01, 20")

    def test_dateTupleToMMMSpaceDDCommaSpaceYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToMMMSpaceDDCommaSpaceYYYY(dateTuple), "AUG 01, 2020")

    def test_dateTupleToDDDashMMDashYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToDDDashMMDashYYYY(dateTuple), "01-08-2020")

    def test_dateTupleToDDashMDashYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToDDashMDashYYYY(dateTuple), "1-8-2020")

    def test_dateTupleToDDDotMMDotYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToDDDotMMDotYYYY(dateTuple), "01.08.2020")

    def test_dateTupleToDDotMDotYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToDDotMDotYYYY(dateTuple), "1.8.2020")

    def test_dateTupleToYYYYDotMMDotDD(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToYYYYDotMMDotDD(dateTuple), "2020.08.01")

    def test_dateTupleToYYYYDotMDotD(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToYYYYDotMDotD(dateTuple), "2020.8.1")


    def test_dateTupleToYYYYMMDD(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToYYYYMMDD(dateTuple), "20200801")

    def test_dateTupleToYYYYDashMDashD(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToYYYYDashMDashD(dateTuple), "20200801")

    def test_dateTupleToDSpaceMMMSpaceYYYY(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToDSpaceMMMSpaceYYYY(dateTuple), "1 AUG 2020")

    def test_dateTupleToYYYYDashMMDashDD(self):
        dateTuple = [2020,8,1]
        self.assertEqual(generator.dateTupleToYYYYDashMMDashDD(dateTuple), "2020-08-01")



if __name__ == '__main__':
    unittest.main()
