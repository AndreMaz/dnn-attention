import dataset.generator

minYear = '1950-01-01'
maxYear = '2050-12-31'


def main(minYear: int, maxYear: int) -> None:
    a = dataset.generator.generate(minYear, maxYear)
    print(a[5000])
    print(dataset.generator.first3Letters(dataset.generator.MONTH_NAMES_FULL))
   # a = dataset.generator.first3Letters('ABCDE')
   # print(a)


main(minYear, maxYear)
