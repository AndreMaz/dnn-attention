import sys
sys.path.append('.')

import unittest
from dataset import date_format
import numpy as np

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
        result = date_format.first3Letters(date_format.MONTH_NAMES_FULL)
        self.assertEqual(len(result), 12)
        self.assertEqual(result[0], "JAN")
        self.assertEqual(result[1], "FEB")

    def test_uniqueMonthLetters(self):
        monthList = date_format.first3Letters(date_format.MONTH_NAMES_FULL)
        result = date_format.uniqueMonthLetters(monthList)
        self.assertEqual(result, "ABCDEFGJLMNOPRSTUVY")

    def test_InputVocab(self):
        self.assertEqual(len(date_format.INPUT_VOCAB), 35)

    def test_OutputVocab(self):
        self.assertEqual(len(date_format.OUTPUT_VOCAB), 13)

    def test_toTwoDigitString(self):
        self.assertEqual(date_format.toTwoDigitString(2), "02")
        self.assertEqual(date_format.toTwoDigitString(11), "11")

    def test_dateTupleToDDMMMYYYY(self):
        dateTuple = [2020, 12, 30]
        self.assertEqual(date_format.dateTupleToDDMMMYYYY(
            dateTuple), "30DEC2020")

    def test_dateTupleToMMSlashDDSlashYYYY(self):
        dateTuple = [2020, 12, 30]
        self.assertEqual(date_format.dateTupleToMMSlashDDSlashYYYY(
            dateTuple), "12/30/2020")

    def test_dateTupleToMSlashDSlashYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(
            date_format.dateTupleToMSlashDSlashYYYY(dateTuple), "8/1/2020")

    def test_dateTupleToMMSlashDDSlashYY(self):
        dateTuple = [2021, 8, 30]
        self.assertEqual(
            date_format.dateTupleToMMSlashDDSlashYY(dateTuple), "08/30/21")

    def test_dateTupleToMSlashDSlashYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(
            date_format.dateTupleToMSlashDSlashYY(dateTuple), "8/1/20")

    def test_dateTupleToMMDDYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToMMDDYY(dateTuple), "080120")

    def test_dateTupleToMMMSpaceDDSpaceYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToMMMSpaceDDSpaceYY(
            dateTuple), "AUG 01 20")

    def test_dateTupleToMMMSpaceDDSpaceYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToMMMSpaceDDSpaceYYYY(
            dateTuple), "AUG 01 2020")

    def test_dateTupleToMMMSpaceDDCommaSpaceYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToMMMSpaceDDCommaSpaceYY(
            dateTuple), "AUG 01, 20")

    def test_dateTupleToMMMSpaceDDCommaSpaceYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToMMMSpaceDDCommaSpaceYYYY(
            dateTuple), "AUG 01, 2020")

    def test_dateTupleToDDDashMMDashYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToDDDashMMDashYYYY(
            dateTuple), "01-08-2020")

    def test_dateTupleToDDashMDashYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(
            date_format.dateTupleToDDashMDashYYYY(dateTuple), "1-8-2020")

    def test_dateTupleToDDDotMMDotYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToDDDotMMDotYYYY(
            dateTuple), "01.08.2020")

    def test_dateTupleToDDotMDotYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(
            date_format.dateTupleToDDotMDotYYYY(dateTuple), "1.8.2020")

    def test_dateTupleToYYYYDotMMDotDD(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToYYYYDotMMDotDD(
            dateTuple), "2020.08.01")

    def test_dateTupleToYYYYDotMDotD(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(
            date_format.dateTupleToYYYYDotMDotD(dateTuple), "2020.8.1")

    def test_dateTupleToYYYYMMDD(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(
            date_format.dateTupleToYYYYMMDD(dateTuple), "20200801")

    def test_dateTupleToYYYYDashMDashD(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(
            date_format.dateTupleToYYYYDashMDashD(dateTuple), "20200801")

    def test_dateTupleToDSpaceMMMSpaceYYYY(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToDSpaceMMMSpaceYYYY(
            dateTuple), "1 AUG 2020")

    def test_dateTupleToYYYYDashMMDashDD(self):
        dateTuple = [2020, 8, 1]
        self.assertEqual(date_format.dateTupleToYYYYDashMMDashDD(
            dateTuple), "2020-08-01")

    def test_InputFormats(self):
        self.assertEqual(len(date_format.INPUT_FNS), 20)

    def test_encodeInputDateStrings(self):
        dateStrings = ["1 AUG 2020"]
        actual = date_format.encodeInputDateStrings(dateStrings)
        expected = [[2., 15., 16., 32., 22., 15., 3., 1., 3., 1., 0., 0.]]

        self.assertTrue(np.all(actual == expected))

    def test_encodeInputDateStrings_fail(self):
        dateStrings = ["1 not in vocabulary 2020  "]

        with self.assertRaises(ValueError):
            date_format.encodeInputDateStrings(dateStrings)

    def test_encodeOutputDateStrings(self):
        dateStrings = ["2050-12-31"]
        actual = date_format.encodeOutputDateStrings(dateStrings)
        expected = [[4, 2, 7, 2, 12, 3, 4, 12, 5, 3]]

        self.assertTrue(np.all(actual == expected))

    def test_encodeOutputDateStrings_fail(self):
        dateStrings = ["not in vocabulary-12-31"]

        with self.assertRaises(ValueError):
            date_format.encodeOutputDateStrings(dateStrings)


if __name__ == '__main__':
    unittest.main()
