import pandas as pd

def generate(minYear: int, maxYear: int) -> list:
    daterange = pd.date_range(minYear, maxYear)
    dates = []
    for single_date in daterange:
        date: list = single_date.strftime("%Y-%m-%d").split('-')

        for index, value in enumerate(date):
            date[index] = int(date[index])

        dates.append(date)

    return dates

