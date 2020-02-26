# import dataset.generator
from dataset import generator

minYear = '1950-01-01'
maxYear = '2050-12-31'
dateTuple = [2050,12,31]

def main(minYear: str, maxYear: str) -> None:
    a = generator.generate(minYear, maxYear)
    print(a[5000])
    
    monthList = generator.first3Letters(generator.MONTH_NAMES_FULL)
    print(monthList)
    print(generator.uniqueMonthLetters(monthList))
    print(generator.INPUT_VOCAB)    

    print(generator.dateTupleToDDMMMYYYY(dateTuple))

main(minYear, maxYear)
