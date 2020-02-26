# import dataset.generator
from dataset import date_format

minYear = '1950-01-01'
maxYear = '2050-12-31'


def main(minYear: str, maxYear: str) -> None:
    # a = date_format.generate(minYear, maxYear)
    # print(a[5000])
    
    # monthList = date_format.first3Letters(date_format.MONTH_NAMES_FULL)
    # print(monthList)
    # print(date_format.uniqueMonthLetters(monthList))
    # print(date_format.INPUT_VOCAB)    

    # dateTuple = [2050,12,31]
    # print(date_format.dateTupleToDDMMMYYYY(dateTuple))
    dateStrings = ["1 AUG 2020  "]
    print(date_format.encodeInputDateStrings(dateStrings))

    dateStrings = ["2050-12-31"]
    print(date_format.encodeOutputDateStrings(dateStrings))

main(minYear, maxYear)
